import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from celery import Celery
from celery.schedules import schedule
from fakeredis import FakeRedis
from redis.exceptions import WatchError

from redbeat.decoder import RedBeatJSONEncoder
from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_integration import app as redis_app  # noqa: F401 - registers Redis fixture

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
FINGERPRINT = 'test-policy-v1'


@pytest.fixture(params=['memory', 'redis'])
def app(request):
    if request.param == 'redis':
        result = request.getfixturevalue('redis_app')
        result.now = lambda: NOW
        yield result
        return
    result = Celery('jitter-concurrent-tests', broker='memory://', set_as_current=False)
    result.conf.update(redbeat_redis_url='redis://unused/0', timezone='UTC')
    result.redbeat_redis = FakeRedis(decode_responses=True)
    result.now = lambda: NOW
    yield result
    result.close()


def create_entry(app, due=NOW, **kwargs):
    """
    Store a generic managed interval entry using the isolated Redis fixture.

    :param app: Isolated Celery app
    :param due: Exact initial due time
    :param kwargs: Optional entry fields

    :return: Saved entry
    """
    initial = RedBeatSchedulerEntry(
        'concurrent-entry', 'tasks.test', schedule(60, app=app), app=app, **kwargs
    )
    return initial.save_with_first_due(due, FINGERPRINT)


def load(entry):
    return RedBeatSchedulerEntry.from_key(entry.key, app=entry.app)


def test_reservation_preserves_latest_payload_and_entry_fields(app):
    stale = create_entry(app, kwargs={'value': 'old'}, options={'expires': 60})
    current = load(stale)
    current.kwargs = {'value': 'new'}
    current.args = ['new-argument']
    current.options = {'expires': 120, 'queue': 'new-queue'}
    current.enabled = False
    current.update_definition()
    current_definition = app.redbeat_redis.hget(stale.key, 'definition')

    reserved = next(stale)
    stored = load(stale)

    assert stored.kwargs == {'value': 'new'}
    assert stored.args == ['new-argument']
    assert stored.options == current.options
    assert not stored._strtobool(stored.enabled)
    # Reservation retains the already-loaded invocation. It cannot cancel an
    # in-flight publication, but must not revert the configuration for future runs.
    assert reserved.kwargs == {'value': 'old'}
    assert reserved.args is None
    assert reserved.options == {'expires': 60}
    assert reserved._strtobool(reserved.enabled)
    assert stored.last_run_at == NOW
    assert stored.total_run_count == 1
    assert app.redbeat_redis.hget(stale.key, 'definition') == current_definition
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key)
        == (NOW + timedelta(seconds=60)).timestamp()
    )


def test_reservation_does_not_resurrect_deleted_managed_entry(app):
    stale = create_entry(app)
    stale.delete()

    with pytest.raises(KeyError):
        next(stale)

    assert not app.redbeat_redis.exists(stale.key)
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key) is None


def test_reservation_does_not_overwrite_recreated_entry_with_same_policy(app):
    stale = create_entry(app)
    stale.delete()
    replacement = create_entry(app, NOW + timedelta(seconds=30), kwargs={'value': 'replacement'})
    before = app.redbeat_redis.hgetall(replacement.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, replacement.key)

    with pytest.raises(ValueError, match='run state'):
        next(stale)

    assert app.redbeat_redis.hgetall(replacement.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, replacement.key) == score


def test_reservation_preserves_new_immediate_request_after_entry_was_loaded(app):
    stale = create_entry(app)
    current = load(stale)
    current.run_immediately = True
    current.save()
    requested_score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key)

    next(stale)
    stored = load(stale)

    assert stored._strtobool(stored.run_immediately)
    assert app.redbeat_redis.sismember('run_immediately', stale.key)
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key) == requested_score

    # The newer request is served on the next selection, then cleared normally.
    app.now = lambda: NOW + timedelta(seconds=1)
    assert stored.is_due()[0]
    next(stored)
    finished = load(stale)
    assert not finished._strtobool(finished.run_immediately)
    assert not app.redbeat_redis.sismember('run_immediately', stale.key)
    assert finished.total_run_count == 2
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key)
        == (NOW + timedelta(seconds=61)).timestamp()
    )


def test_stale_reservation_cannot_replace_a_completed_reservation(app):
    initial = create_entry(app)
    stale = load(initial)
    next(initial)
    before = app.redbeat_redis.hgetall(initial.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key)
    with pytest.raises(ValueError, match='run state'):
        next(stale)
    assert app.redbeat_redis.hgetall(initial.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == score


@pytest.mark.parametrize(
    'change,error,message',
    [
        ('deleted', KeyError, None),
        ('recreated', ValueError, 'The schedule run state changed before reservation'),
        ('policy', ValueError, 'The applied jitter policy changed'),
        ('interval', ValueError, 'The schedule interval changed before reservation'),
        ('run-state', ValueError, 'The schedule run state changed before reservation'),
    ],
)
def test_invalid_reservation_state_is_not_written_or_retried(app, change, error, message):
    stale = create_entry(app)
    if change == 'deleted':
        stale.delete()
    elif change == 'recreated':
        stale.delete()
        create_entry(app, NOW + timedelta(seconds=30))
    elif change == 'policy':
        app.redbeat_redis.hset(stale.key, 'jitter_fingerprint', 'changed-policy')
    elif change == 'interval':
        current = load(stale)
        current.schedule = schedule(120, app=app)
        current.save()
    else:
        next(load(stale))

    before = app.redbeat_redis.hgetall(stale.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key)
    immediate = app.redbeat_redis.sismember('run_immediately', stale.key)
    pipeline_class = type(app.redbeat_redis.pipeline())
    original_watch = pipeline_class.watch
    watch_calls = []

    def record_watch(pipe, *keys):
        watch_calls.append(keys)
        return original_watch(pipe, *keys)

    with (
        patch.object(pipeline_class, 'watch', record_watch),
        patch.object(pipeline_class, 'execute') as execute,
    ):
        with pytest.raises(error, match=message):
            next(stale)
        execute.assert_not_called()

    assert watch_calls == [(stale.key,)]
    assert app.redbeat_redis.hgetall(stale.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, stale.key) == score
    assert app.redbeat_redis.sismember('run_immediately', stale.key) == immediate


def test_definition_update_does_not_restore_consumed_initial_note(app):
    stale = create_entry(
        app, kwargs={'value': 'old'}, first_run_kwargs={'bootstrap_token': 'first'}
    )
    next(load(stale))
    stale.kwargs['value'] = 'new'
    stale.update_definition()

    stored = load(stale)
    assert stored.kwargs == {'value': 'new'}
    assert stored.first_run_kwargs == {}
    assert stored.total_run_count == 1
    assert stored.last_run_at == NOW


def test_definition_update_retries_metadata_watch_conflict(app):
    current = create_entry(app, kwargs={'value': 'old'})
    current.kwargs = {'value': 'new'}
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    writes = []

    def conflict_once(pipe, *args, **kwargs):
        if not writes:
            writes.append(True)
            app.redbeat_redis.hset(
                current.key,
                'meta',
                json.dumps({'last_run_at': NOW, 'total_run_count': 8}, cls=RedBeatJSONEncoder),
            )
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict_once):
        current.update_definition()

    stored = load(current)
    assert writes == [True]
    assert stored.kwargs == {'value': 'new'}
    assert stored.total_run_count == 8
    assert stored.last_run_at == NOW


def test_definition_update_preserves_latest_first_run_kwargs(app):
    stale = create_entry(app, first_run_kwargs={'bootstrap_token': 'old'})
    current = load(stale)
    current.first_run_kwargs = {'bootstrap_token': 'new', 'other': 2}
    current.save()
    stale.kwargs = {'payload': 'updated'}

    stale.update_definition()
    stored = load(stale)

    assert stored.kwargs == {'payload': 'updated'}
    assert stored.first_run_kwargs == {'bootstrap_token': 'new', 'other': 2}


def test_reservation_does_not_consume_newer_first_run_kwargs(app):
    stale = create_entry(app, first_run_kwargs={'bootstrap_token': 'old'})
    current = load(stale)
    current.first_run_kwargs = {'bootstrap_token': 'new'}
    current.save()
    before = app.redbeat_redis.hget(stale.key, 'definition')

    outgoing = next(stale)
    stored = load(stale)

    assert outgoing.kwargs == {'bootstrap_token': 'old'}
    assert stored.first_run_kwargs == {'bootstrap_token': 'new'}
    assert app.redbeat_redis.hget(stale.key, 'definition') == before
    assert next(stored).kwargs == {'bootstrap_token': 'new'}
    assert load(stale).first_run_kwargs == {}


def test_reservation_retries_payload_watch_conflict(app):
    stale = create_entry(app, kwargs={'value': 'old'})
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    writes = []

    def conflict_once(pipe, *args, **kwargs):
        if not writes:
            writes.append(True)
            definition = stale._jitter_definition()
            definition['kwargs'] = {'value': 'new'}
            definition['options'] = {'expires': 120, 'queue': 'new-queue'}
            app.redbeat_redis.hset(
                stale.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder)
            )
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict_once):
        reserved = next(stale)

    stored = load(stale)
    assert writes == [True]
    assert stored.kwargs == {'value': 'new'}
    assert stored.options == {'expires': 120, 'queue': 'new-queue'}
    assert reserved.kwargs == {'value': 'old'}
    assert reserved.options == {}
    assert stored.total_run_count == 1
    assert stored.last_run_at == NOW


@pytest.mark.parametrize('operation', ['definition', 'reservation'])
def test_managed_write_stops_after_three_watch_conflicts(app, operation):
    initial = create_entry(app, kwargs={'value': 'old'})
    before = app.redbeat_redis.hgetall(initial.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key)
    initial.kwargs = {'value': 'new'}
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    attempts = []

    def conflict_every_time(pipe, *args, **kwargs):
        attempts.append(True)
        app.redbeat_redis.hset(initial.key, 'concurrent_value', str(len(attempts)))
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict_every_time):
        with pytest.raises(WatchError):
            if operation == 'definition':
                initial.update_definition()
            else:
                next(initial)

    assert len(attempts) == 3
    after = app.redbeat_redis.hgetall(initial.key)
    assert {key: value for key, value in after.items() if key != 'concurrent_value'} == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == score


def test_exhausted_immediate_reservation_retains_existing_selection_limit(app):
    initial = RedBeatSchedulerEntry(
        'concurrent-immediate', 'tasks.test', schedule(1800, app=app), app=app
    ).save_with_first_due(NOW + timedelta(seconds=1800), FINGERPRINT)
    initial.run_immediately = True
    initial.save()
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key)
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    attempts = []

    def conflict_reservation(pipe, *args, **kwargs):
        reserving = any(
            command[0][:3] == ('HSET', initial.key, 'meta') for command in pipe.command_stack
        )
        if reserving:
            attempts.append(True)
            app.redbeat_redis.hset(initial.key, 'concurrent_value', str(len(attempts)))
        return execute(pipe, *args, **kwargs)

    scheduler = RedBeatScheduler(app, lazy=True, max_interval=1)
    try:
        with patch.object(app, 'send_task') as publish:
            with patch.object(pipeline_class, 'execute', conflict_reservation):
                scheduler.tick()

            assert len(attempts) == 3
            publish.assert_not_called()
            assert not app.redbeat_redis.sismember('run_immediately', initial.key)
            assert load(initial)._strtobool(load(initial).run_immediately)
            assert load(initial).total_run_count == 0
            assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == score
            # Selection already drained the immediate set. Without a separate
            # recovery mechanism, this request waits for normal due selection.
            scheduler.tick()
            publish.assert_not_called()
            app.now = lambda: NOW + timedelta(seconds=1800)
            scheduler.tick()
            publish.assert_called_once()
            stored = load(initial)
            assert stored.total_run_count == 1
            assert not stored._strtobool(stored.run_immediately)
    finally:
        scheduler.close()


def test_tick_does_not_create_immediate_request_for_failed_normal_reservation(app):
    initial = create_entry(app)
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    attempts = []

    def conflict_reservation(pipe, *args, **kwargs):
        reserving = any(
            command[0][:3] == ('HSET', initial.key, 'meta') for command in pipe.command_stack
        )
        if reserving:
            attempts.append(True)
            app.redbeat_redis.hset(initial.key, 'concurrent_value', str(len(attempts)))
        return execute(pipe, *args, **kwargs)

    scheduler = RedBeatScheduler(app, lazy=True, max_interval=1)
    try:
        with patch.object(app, 'send_task') as publish:
            with patch.object(pipeline_class, 'execute', conflict_reservation):
                scheduler.tick()
            assert len(attempts) == 3
            publish.assert_not_called()
            assert not app.redbeat_redis.sismember('run_immediately', initial.key)
            assert not load(initial)._strtobool(load(initial).run_immediately)
            assert load(initial).total_run_count == 0
            scheduler.tick()
            publish.assert_called_once()
    finally:
        scheduler.close()
