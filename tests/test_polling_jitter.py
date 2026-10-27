import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from celery import Celery
from celery.schedules import crontab, schedule
from fakeredis import FakeRedis

from redbeat.decoder import RedBeatJSONDecoder, to_timestamp
from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
FINGERPRINT = 'test-policy-v1'


@pytest.fixture
def app():
    result = Celery('jitter-tests', broker='memory://')
    result.conf.update(redbeat_redis_url='redis://unused/0', timezone='UTC')
    result.redbeat_redis = FakeRedis(decode_responses=True)
    result.now = lambda: NOW
    yield result
    result.close()


def entry(app, interval=60, **kwargs):
    return RedBeatSchedulerEntry(
        'first-due-test', 'tasks.test', schedule(interval, app=app), app=app, **kwargs
    )


@pytest.mark.parametrize('interval,first_delay', [(60, 102), (300, 356), (600, 22), (1800, 35)])
def test_exact_first_due_roundtrip(app, interval, first_delay):
    initial = entry(app, interval, total_run_count=7, run_immediately=True)
    due = NOW + timedelta(seconds=first_delay)
    # A stale immediate-set member must not bypass the accepted first due.
    app.redbeat_redis.sadd('run_immediately', initial.key)
    initial.save_with_first_due(due, FINGERPRINT)
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)
    raw = app.redbeat_redis.hgetall(initial.key)
    assert set(raw) == {'definition', 'meta', 'jitter_fingerprint'}
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == to_timestamp(due)
    assert loaded.last_run_at == due - timedelta(seconds=interval)
    assert loaded.total_run_count == 7
    assert loaded.jitter_fingerprint == FINGERPRINT
    assert loaded.run_immediately == 'False'
    assert not app.redbeat_redis.sismember('run_immediately', initial.key)
    assert loaded.is_due()[0] is False
    app.now = lambda: due - timedelta(microseconds=1)
    assert loaded.is_due()[0] is False
    app.now = lambda: due
    assert loaded.is_due()[0] is True


def test_writes_are_grouped_in_a_transaction(app):
    pipeline_class = type(app.redbeat_redis.pipeline())
    original = pipeline_class.execute
    recorded = []

    def record(pipe, *args, **kwargs):
        recorded.append((pipe.transaction, list(pipe.command_stack)))
        return original(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', record):
        entry(app).save_with_first_due(NOW + timedelta(seconds=42), FINGERPRINT)
    transaction, commands = recorded[0]
    assert transaction is True
    assert [command[0][0] for command in commands] == ['HSET', 'HSET', 'HSET', 'ZADD', 'SREM']
    assert [command[0][2] for command in commands[:3]] == [
        'definition',
        'meta',
        'jitter_fingerprint',
    ]


@pytest.mark.parametrize('state', ['legacy', 'ordinary', 'initial', 'immediate', 'both'])
def test_reservation_field_writes_remain_in_one_transaction(app, state):
    initial = entry(app)
    if state != 'legacy':
        initial.save_with_first_due(
            NOW,
            FINGERPRINT,
            first_run_kwargs={'bootstrap_token': 'first'} if state in ('initial', 'both') else {},
        )
    else:
        initial.save()
    if state in ('immediate', 'both'):
        initial.run_immediately = True
        initial.save()
    pipeline_class = type(app.redbeat_redis.pipeline())
    original = pipeline_class.execute
    recorded = []

    def record(pipe, *args, **kwargs):
        recorded.append((pipe.transaction, list(pipe.command_stack)))
        return original(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', record):
        next(initial)
    transaction, commands = recorded[0]
    assert transaction is True
    changes_definition = state in ('initial', 'immediate', 'both')
    expected = ['HSET', 'HSET'] if changes_definition else ['HSET']
    expected += ['ZADD', 'SREM'] if state != 'legacy' else ['ZADD']
    assert [command[0][0] for command in commands] == expected
    fields = [command[0][2] for command in commands if command[0][0] == 'HSET']
    assert fields == (['meta', 'definition'] if changes_definition else ['meta'])


def test_ordinary_managed_reservation_does_not_rewrite_definition(app):
    initial = entry(app, kwargs={'value': 'unchanged'}).save_with_first_due(NOW, FINGERPRINT)
    definition = json.loads(app.redbeat_redis.hget(initial.key, 'definition'))
    # Preserve the exact stored representation, not only equivalent JSON values.
    raw_definition = json.dumps(definition, indent=2)
    app.redbeat_redis.hset(initial.key, 'definition', raw_definition)

    next(RedBeatSchedulerEntry.from_key(initial.key, app))

    assert app.redbeat_redis.hget(initial.key, 'definition') == raw_definition
    stored = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert stored.last_run_at == NOW
    assert stored.total_run_count == 1
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == to_timestamp(
        NOW + timedelta(seconds=60)
    )


def test_reservation_keeps_existing_false_flag_normalization(app):
    initial = entry(app).save_with_first_due(NOW, FINGERPRINT)
    initial.run_immediately = False
    initial.save()

    next(RedBeatSchedulerEntry.from_key(initial.key, app))

    assert RedBeatSchedulerEntry.from_key(initial.key, app).run_immediately == 'False'


@pytest.mark.parametrize(
    'due',
    [
        NOW.replace(tzinfo=None),
        NOW.replace(microsecond=1),
        NOW.astimezone(timezone(timedelta(hours=1))),
        None,
    ],
)
def test_reject_invalid_due_without_writes(app, due):
    initial = entry(app)
    with pytest.raises(ValueError, match='UTC'):
        initial.save_with_first_due(due, FINGERPRINT)
    assert app.redbeat_redis.dbsize() == 0


@pytest.mark.parametrize('interval', [0, -1, 1.5])
def test_reject_invalid_interval_without_writes(app, interval):
    with pytest.raises(ValueError, match='interval'):
        entry(app, interval).save_with_first_due(NOW, FINGERPRINT)
    assert app.redbeat_redis.dbsize() == 0


@pytest.mark.parametrize('fingerprint', [None, '', False, 1])
def test_reject_invalid_fingerprint_without_writes(app, fingerprint):
    initial = entry(app)
    with pytest.raises(ValueError, match='fingerprint'):
        initial.save_with_first_due(NOW, fingerprint)
    assert app.redbeat_redis.dbsize() == 0


def test_reject_noninterval_and_relative_schedule(app):
    for value in (crontab(), schedule(60, relative=True, app=app)):
        initial = entry(app)
        initial.schedule = value
        with pytest.raises(ValueError):
            initial.save_with_first_due(NOW, FINGERPRINT)


def test_existing_entry_not_overwritten(app):
    initial = entry(app).save_with_first_due(NOW + timedelta(seconds=20), FINGERPRINT)
    before = app.redbeat_redis.hgetall(initial.key)
    with pytest.raises(ValueError):
        initial.save_with_first_due(NOW + timedelta(seconds=50), FINGERPRINT)
    assert app.redbeat_redis.hgetall(initial.key) == before


def test_first_due_refuses_orphan_hash_instead_of_repairing_it(app):
    initial = entry(app)
    app.redbeat_redis.hset(initial.key, 'meta', '{}')
    before = app.redbeat_redis.hgetall(initial.key)
    with pytest.raises(ValueError, match='existing schedule hash'):
        initial.save_with_first_due(NOW, FINGERPRINT)
    assert app.redbeat_redis.hgetall(initial.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) is None


def test_definition_only_preserves_due_metadata_and_extra_hash_fields(app):
    initial = entry(
        app,
        kwargs={'address': 'router-01:161'},
        options={'expires': 60, 'link': ['callback']},
    )
    initial.save_with_first_due(
        NOW + timedelta(seconds=20), FINGERPRINT, first_run_kwargs={'bootstrap_token': 'first'}
    )
    app.redbeat_redis.hset(initial.key, 'custom_metadata', 'preserve-me')
    before = app.redbeat_redis.hgetall(initial.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key)
    initial.kwargs = {'address': 'router-01:161', 'profile': 'changed'}
    initial.run_immediately = True  # Definition-only must preserve stored idle state.
    initial.update_definition()
    after = app.redbeat_redis.hgetall(initial.key)
    assert {key: value for key, value in before.items() if key != 'definition'} == {
        key: value for key, value in after.items() if key != 'definition'
    }
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == score
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert loaded.kwargs['profile'] == 'changed'
    assert loaded.first_run_kwargs == {'bootstrap_token': 'first'}
    assert 'bootstrap_token' not in loaded.kwargs
    assert loaded.options == {'expires': 60, 'link': ['callback']}
    assert loaded.run_immediately == 'False'


def test_definition_update_cannot_recreate_or_rephase(app):
    initial = entry(app).save_with_first_due(NOW + timedelta(seconds=20), FINGERPRINT)
    initial.schedule = schedule(120, app=app)
    with pytest.raises(ValueError, match='interval'):
        initial.update_definition()
    initial.delete()
    with pytest.raises(KeyError):
        initial.update_definition()
    assert not app.redbeat_redis.exists(initial.key)


@pytest.mark.parametrize('fingerprint', [None, '', 'changed-policy'])
def test_definition_update_refuses_missing_or_changed_policy(app, fingerprint):
    initial = entry(app).save_with_first_due(NOW + timedelta(seconds=20), FINGERPRINT)
    if fingerprint is None:
        app.redbeat_redis.hdel(initial.key, 'jitter_fingerprint')
    else:
        app.redbeat_redis.hset(initial.key, 'jitter_fingerprint', fingerprint)
    before = app.redbeat_redis.hgetall(initial.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key)
    initial.kwargs = {'updated': True}
    with pytest.raises(ValueError, match='policy changed'):
        initial.update_definition()
    assert app.redbeat_redis.hgetall(initial.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == score


def test_reservation_keeps_outgoing_context_only(app):
    initial = entry(app, kwargs={'address': 'router-01:161'})
    due = NOW + timedelta(seconds=20)
    initial.save_with_first_due(due, FINGERPRINT, first_run_kwargs={'bootstrap_token': 'first'})
    copied = RedBeatSchedulerEntry(**dict(initial))
    assert copied.jitter_fingerprint == FINGERPRINT
    app.now = lambda: due
    scheduler = RedBeatScheduler(app, lazy=True)
    outgoing = scheduler.reserve(initial)
    stored = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert outgoing.kwargs['bootstrap_token'] == 'first'
    assert outgoing.first_run_kwargs == {}
    assert stored.first_run_kwargs == {}
    assert 'bootstrap_token' not in stored.kwargs
    assert initial.first_run_kwargs == {'bootstrap_token': 'first'}
    assert 'bootstrap_token' not in initial.kwargs
    assert outgoing.kwargs is not initial.kwargs
    assert stored.last_run_at == due
    assert stored.total_run_count == 1
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stored.key) == to_timestamp(
        due + timedelta(seconds=60)
    )
    app.now = lambda: due + timedelta(seconds=60)
    assert stored.is_due()[0]
    assert 'bootstrap_token' not in next(stored).kwargs


def test_only_update_last_run_preserves_count(app):
    initial = entry(app, total_run_count=5).save_with_first_due(
        NOW + timedelta(seconds=20), FINGERPRINT
    )
    assert initial.next(last_run_at=NOW, only_update_last_run_at=True).total_run_count == 5


def test_managed_reservation_clears_consumed_immediate_request(app):
    initial = entry(app).save_with_first_due(NOW + timedelta(seconds=20), FINGERPRINT)
    initial.set_run_immediately(True)
    initial.save()
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert loaded.is_due()[0]
    outgoing = next(loaded)
    stored = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert outgoing.last_run_at == stored.last_run_at == NOW
    assert stored.total_run_count == 1
    assert stored.run_immediately == 'False'
    assert not app.redbeat_redis.sismember('run_immediately', initial.key)
    assert stored.jitter_fingerprint == FINGERPRINT
    app.now = lambda: NOW + timedelta(seconds=1)
    assert not stored.is_due()[0]


def test_legacy_entries_keep_ordinary_definition(app):
    initial = entry(app).save()
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert loaded.jitter_fingerprint is None
    assert 'jitter_fingerprint' not in json.loads(
        app.redbeat_redis.hget(initial.key, 'definition'), cls=RedBeatJSONDecoder
    )
    assert not app.redbeat_redis.hexists(initial.key, 'jitter_fingerprint')
    definition = app.redbeat_redis.hget(initial.key, 'definition')
    app.redbeat_redis.sadd('run_immediately', initial.key)
    next(loaded)
    assert not app.redbeat_redis.hexists(initial.key, 'jitter_fingerprint')
    assert app.redbeat_redis.hget(initial.key, 'definition') == definition
    assert app.redbeat_redis.sismember('run_immediately', initial.key)


def test_first_run_kwargs_override_only_the_first_invocation(app):
    initial = entry(app, kwargs={'mode': 'regular', 'value': 7}).save_with_first_due(
        NOW, FINGERPRINT, first_run_kwargs={'mode': 'bootstrap', 'token': 'first'}
    )
    outgoing = next(initial)
    stored = RedBeatSchedulerEntry.from_key(initial.key, app)

    assert outgoing.kwargs == {'mode': 'bootstrap', 'value': 7, 'token': 'first'}
    assert stored.kwargs == {'mode': 'regular', 'value': 7}
    assert stored.first_run_kwargs == {}
    assert next(stored).kwargs == {'mode': 'regular', 'value': 7}


@pytest.mark.parametrize(
    'supplied,expected',
    [
        (None, {'constructor_token': 'first'}),
        ({'replacement_token': 'first'}, {'replacement_token': 'first'}),
        ({}, {}),
    ],
)
def test_first_due_selects_first_run_kwargs_without_changing_recurring_kwargs(
    app, supplied, expected
):
    initial = entry(
        app,
        kwargs={'permanent': 'keep'},
        first_run_kwargs={'constructor_token': 'first'},
    ).save_with_first_due(NOW, FINGERPRINT, first_run_kwargs=supplied)

    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert loaded.first_run_kwargs == expected
    assert loaded.kwargs == {'permanent': 'keep'}
    assert next(loaded).kwargs == {'permanent': 'keep', **expected}
    assert RedBeatSchedulerEntry.from_key(initial.key, app).first_run_kwargs == {}


def test_managed_operations_use_subclass_definition_decoder(app):
    decoded = []

    class CustomEntry(RedBeatSchedulerEntry):
        @staticmethod
        def decode_definition(definition):
            decoded.append(definition)
            return RedBeatSchedulerEntry.decode_definition(definition)

    initial = CustomEntry(
        'custom-decoder',
        'tasks.test',
        schedule(60, app=app),
        app=app,
        kwargs={'value': 'old'},
    ).save_with_first_due(NOW, FINGERPRINT, first_run_kwargs={'bootstrap_token': 'first'})

    loaded = CustomEntry.from_key(initial.key, app)
    assert len(decoded) == 1
    loaded.kwargs = {'value': 'new'}
    loaded.update_definition()
    assert len(decoded) == 2
    assert next(loaded).kwargs == {'value': 'new', 'bootstrap_token': 'first'}
    assert len(decoded) == 3
    stored = json.loads(app.redbeat_redis.hget(initial.key, 'definition'))
    assert stored['kwargs'] == {'value': 'new'}
    assert 'first_run_kwargs' not in stored


@pytest.mark.parametrize('operation', ['save', 'reschedule'])
def test_first_run_kwargs_survive_existing_field_write_methods(app, operation):
    initial = entry(app, first_run_kwargs={'bootstrap_token': 'first'})
    getattr(initial, operation)()
    stored = RedBeatSchedulerEntry.from_key(initial.key, app)

    assert stored.first_run_kwargs == {'bootstrap_token': 'first'}
    assert stored.kwargs == {}


def test_first_run_kwargs_can_be_cancelled_before_immediate_run(app):
    initial = entry(app).save_with_first_due(
        NOW + timedelta(seconds=20), FINGERPRINT, first_run_kwargs={'bootstrap_token': 'first'}
    )
    initial.first_run_kwargs = {}
    initial.run_immediately = True
    initial.save()
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)

    assert loaded.first_run_kwargs == {}
    assert next(loaded).kwargs == {}
    assert 'first_run_kwargs' not in json.loads(app.redbeat_redis.hget(initial.key, 'definition'))


def test_recurring_kwargs_are_not_implicitly_first_run_kwargs(app):
    initial = entry(app, kwargs={'bootstrap_token': 'permanent'}).save_with_first_due(
        NOW, FINGERPRINT
    )
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)

    assert loaded.first_run_kwargs == {}
    assert next(loaded).kwargs == {'bootstrap_token': 'permanent'}
    assert RedBeatSchedulerEntry.from_key(initial.key, app).kwargs == {
        'bootstrap_token': 'permanent'
    }


def test_first_run_and_recurring_kwargs_remain_separate(app):
    initial = entry(app, kwargs={'bootstrap_token': 'first'}).save_with_first_due(
        NOW, FINGERPRINT, first_run_kwargs={'other_token': 'once'}
    )
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)

    assert loaded.kwargs == {'bootstrap_token': 'first'}
    assert loaded.first_run_kwargs == {'other_token': 'once'}
    assert next(loaded).kwargs == {'bootstrap_token': 'first', 'other_token': 'once'}
    stored = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert stored.kwargs == {'bootstrap_token': 'first'}
    assert stored.first_run_kwargs == {}
