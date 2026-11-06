import json
from datetime import timedelta
from unittest.mock import patch
from uuid import UUID

import pytest
from celery.beat import ScheduleEntry
from celery.schedules import schedule
from redis.exceptions import WatchError

from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_concurrent_updates import app as lifecycle_app
from tests.test_jitter_concurrent_updates import create_entry, load
from tests.test_jitter_integration import FINGERPRINT, NOW
from tests.test_jitter_integration import app as redis_app  # noqa: F401

app = lifecycle_app


def saved_state(entry):
    client = entry.app.redbeat_redis
    return (
        client.hgetall(entry.key),
        client.zscore(entry.app.redbeat_conf.schedule_key, entry.key),
        client.sismember('run_immediately', entry.key),
    )


def apply_operation(entry, operation):
    if operation == 'definition':
        entry.kwargs = {'payload': 'stale-update'}
        return entry.update_definition()
    if operation == 'request':
        entry.kwargs = {'payload': 'stale-request'}
        return entry.request_run(NOW + timedelta(seconds=20))
    return next(entry)


@pytest.mark.parametrize('operation', ['definition', 'request', 'reservation'])
def test_same_key_replacement_rejects_stale_operation_with_identical_timing(app, operation):
    stale = create_entry(app, kwargs={'payload': 'old'})
    stale.delete()
    replacement = create_entry(app, kwargs={'payload': 'replacement'})
    assert replacement.last_run_at == stale.last_run_at
    assert replacement.total_run_count == stale.total_run_count
    assert replacement.entry_id != stale.entry_id
    before = saved_state(replacement)

    with pytest.raises(KeyError):
        apply_operation(stale, operation)

    assert saved_state(replacement) == before


@pytest.mark.parametrize('operation', ['definition', 'request', 'reservation'])
@pytest.mark.parametrize('changed_state', ['policy', 'interval'])
def test_replacement_with_different_policy_or_interval_still_rejects_stale_operation(
    app, operation, changed_state
):
    stale = create_entry(app)
    stale.delete()
    replacement = create_entry(app)
    if changed_state == 'policy':
        app.redbeat_redis.hset(replacement.key, 'jitter_fingerprint', json.dumps('other-policy'))
    else:
        replacement.schedule = schedule(120, app=app)
        replacement.request_run(NOW, run_immediately=True, fingerprint=FINGERPRINT)
    before = saved_state(replacement)

    with pytest.raises(KeyError):
        apply_operation(stale, operation)

    assert saved_state(replacement) == before


def test_replacement_with_different_run_metadata_still_rejects_stale_reservation(app):
    stale = create_entry(app)
    stale.delete()
    replacement = create_entry(app, NOW + timedelta(seconds=30))
    before = saved_state(replacement)

    with pytest.raises(KeyError):
        next(stale)

    assert saved_state(replacement) == before


@pytest.mark.parametrize('run_immediately', [False, True])
def test_stale_same_key_reservation_preserves_replacements_newer_request(app, run_immediately):
    stale = create_entry(app)
    stale.delete()
    replacement = create_entry(app, kwargs={'payload': 'replacement'})
    due = NOW if run_immediately else NOW + timedelta(seconds=20)
    replacement.request_run(due, run_immediately=run_immediately)
    before = saved_state(replacement)

    with pytest.raises(KeyError):
        next(stale)

    stored = load(replacement)
    assert stored.kwargs == replacement.kwargs
    assert saved_state(replacement) == before
    assert stored.total_run_count == 0
    assert stored.last_run_at == due - timedelta(seconds=60)


def test_second_loaded_reservation_rejects_changed_run_state(app):
    first = create_entry(app)
    second = load(first)
    next(first)
    before = saved_state(first)

    with pytest.raises(ValueError, match='run state'):
        next(second)

    assert saved_state(first) == before
    assert load(first).total_run_count == 1


@pytest.mark.parametrize('remove_policy', [False, True])
def test_following_timing_is_written_with_the_reservation(app, remove_policy):
    entry = load(
        create_entry(
            app,
            kwargs={'payload': 'current'},
            options={'queue': 'custom-queue', 'expires': 120},
            total_run_count=9,
        )
    )
    before = saved_state(entry)
    due = NOW + timedelta(seconds=120)
    fingerprint = None if remove_policy else 'new-policy'
    app.conf.redbeat_entry_policy = lambda following: (due, fingerprint)
    pipeline_class = type(app.redbeat_redis.pipeline())
    original_execute = pipeline_class.execute
    writes = []

    def execute(pipe, *args, **kwargs):
        commands = [command for command, _ in pipe.command_stack]
        if any(command[0] == 'ZADD' for command in commands):
            assert saved_state(entry) == before
            writes.append(commands)
        return original_execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', execute):
        outgoing = next(entry)

    assert len(writes) == 1
    assert [command[0] for command in writes[0]] == [
        'HSET',
        'HDEL' if remove_policy else 'HSET',
        'ZADD',
        'SREM',
    ]

    stored = load(entry)
    assert stored.kwargs == entry.kwargs
    assert stored.options == entry.options
    assert stored.total_run_count == 10
    assert outgoing.last_run_at == NOW
    assert outgoing.total_run_count == 10
    assert stored.last_run_at == due - timedelta(seconds=60)
    assert stored.first_run_kwargs == {}
    assert not stored._strtobool(stored.run_immediately)
    assert not app.redbeat_redis.sismember('run_immediately', entry.key)
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == due.timestamp()
    assert stored.jitter_fingerprint == fingerprint
    if remove_policy:
        assert not app.redbeat_redis.hexists(entry.key, 'jitter_fingerprint')


@pytest.mark.parametrize('due', [None, NOW + timedelta(seconds=120)])
@pytest.mark.parametrize('fingerprint', ['', False, 42])
def test_invalid_following_policy_keeps_normal_reservation(app, due, fingerprint):
    entry = load(create_entry(app))
    app.conf.redbeat_entry_policy = lambda following: (due, fingerprint)
    next(entry)
    stored = load(entry)
    assert stored.last_run_at == NOW
    assert stored.total_run_count == 1
    assert stored.jitter_fingerprint == FINGERPRINT
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
        == (NOW + timedelta(seconds=60)).timestamp()
    )


def test_following_timing_keeps_exact_state_after_repeated_watch_conflicts(app):
    entry = load(create_entry(app))
    before = saved_state(entry)
    app.conf.redbeat_entry_policy = lambda following: (NOW + timedelta(seconds=120), 'new-policy')
    pipeline_class = type(app.redbeat_redis.pipeline())
    original_execute = pipeline_class.execute
    attempts = []

    def execute(pipe, *args, **kwargs):
        if any(command[0] == 'ZADD' for command, _ in pipe.command_stack):
            attempts.append(True)
            app.redbeat_redis.hset(entry.key, 'concurrent_value', str(len(attempts)))
        return original_execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', execute):
        with pytest.raises(WatchError):
            next(entry)

    assert len(attempts) == 3
    after = saved_state(entry)
    assert after[0].pop('concurrent_value') == '3'
    assert after == before


def test_watch_conflict_rejects_identical_timing_replacement_after_reload(app):
    stale = create_entry(app, kwargs={'payload': 'old'})
    pipeline_class = type(app.redbeat_redis.pipeline())
    original_execute = pipeline_class.execute
    replaced = []
    replacement_state = []

    def replace_before_commit(pipe, *args, **kwargs):
        if not replaced:
            replaced.append(True)
            with patch.object(pipeline_class, 'execute', original_execute):
                stale.delete()
                replacement = create_entry(app, kwargs={'payload': 'replacement'})
                replacement_state.append(saved_state(replacement))
        return original_execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', replace_before_commit):
        with pytest.raises(KeyError):
            next(stale)

    assert replaced == [True]
    assert saved_state(stale) == replacement_state[0]
    assert load(stale).total_run_count == 0
    assert load(stale).kwargs == {'payload': 'replacement'}


def test_ordinary_entry_does_not_acquire_managed_state(app):
    entry = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app
    ).save()
    next(load(entry))
    stored = load(entry)
    stored.kwargs = {'payload': 'ordinary'}
    stored.save()

    assert not stored.jitter_fingerprint
    fields = app.redbeat_redis.hgetall(entry.key)
    assert 'jitter_fingerprint' not in fields
    assert 'entry_id' not in fields
    assert stored.entry_id is None
    assert 'entry_id' not in json.loads(fields['definition'])


def test_guarded_creation_assigns_identity_and_preserves_it_during_updates(app):
    entry = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app
    ).save_guarded(create=True)
    assert UUID(entry.entry_id).version == 4
    assert load(entry).entry_id == entry.entry_id
    assert json.loads(app.redbeat_redis.hget(entry.key, 'entry_id')) == entry.entry_id
    assert 'entry_id' not in json.loads(app.redbeat_redis.hget(entry.key, 'definition'))
    entry.kwargs = {'payload': 'updated'}
    updated = entry.save_guarded()
    assert updated.entry_id == entry.entry_id
    assert updated.kwargs == entry.kwargs
    assert updated.jitter_fingerprint is None


def test_identified_ordinary_reservation_keeps_definition_and_normal_interval(app):
    entry = RedBeatSchedulerEntry(
        'ordinary-entry',
        'tasks.test',
        schedule(60, app=app),
        app=app,
        kwargs={'payload': 'current'},
    ).save_guarded(create=True)
    before = app.redbeat_redis.hget(entry.key, 'definition')
    outgoing = next(load(entry))
    stored = load(entry)
    assert stored.entry_id == outgoing.entry_id == entry.entry_id
    assert stored.kwargs == outgoing.kwargs == entry.kwargs
    assert stored.total_run_count == 1
    assert stored.last_run_at == NOW
    assert stored.jitter_fingerprint is None
    assert app.redbeat_redis.hget(entry.key, 'definition') == before
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
        == (NOW + timedelta(seconds=60)).timestamp()
    )


@pytest.mark.parametrize('operation', ['save', 'reservation'])
def test_guarded_ordinary_replacement_rejects_stale_writes(app, operation):
    stale = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app
    ).save_guarded(create=True)
    stale.delete()
    replacement = RedBeatSchedulerEntry(
        stale.name,
        'tasks.test',
        schedule(60, app=app),
        app=app,
        kwargs={'payload': 'replacement'},
    ).save_guarded(create=True)
    before = saved_state(replacement)
    with pytest.raises(KeyError):
        if operation == 'save':
            stale.save_guarded()
        else:
            next(stale)
    assert saved_state(replacement) == before


@pytest.mark.parametrize('identified', [False, True])
def test_guarded_creation_refuses_existing_definition_or_orphan_hash(app, identified):
    entry = RedBeatSchedulerEntry('ordinary-entry', 'tasks.test', schedule(60, app=app), app=app)
    if identified:
        entry.save_guarded(create=True)
    else:
        app.redbeat_redis.hset(entry.key, 'meta', '{}')
    before = saved_state(entry)
    with pytest.raises(ValueError, match='existing schedule hash'):
        entry.save_guarded(create=True)
    assert saved_state(entry) == before


@pytest.mark.parametrize('operation', ['save', 'reservation'])
def test_legacy_none_identity_rejects_new_identified_replacement(app, operation):
    stale = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app
    ).save()
    assert stale.entry_id is None
    stale.delete()
    replacement = RedBeatSchedulerEntry(
        stale.name, 'tasks.test', schedule(60, app=app), app=app
    ).save_guarded(create=True)
    before = saved_state(replacement)
    with pytest.raises(KeyError):
        if operation == 'save':
            stale.save_guarded()
        else:
            app.conf.redbeat_entry_policy = lambda entry: (None, 'new-policy')
            next(stale)
    assert saved_state(replacement) == before


@pytest.mark.parametrize('operation', ['definition', 'request', 'reservation'])
def test_legacy_managed_copy_rejects_identified_replacement(app, operation):
    original = create_entry(app)
    app.redbeat_redis.hdel(original.key, 'entry_id')
    stale = load(original)
    assert stale.entry_id is None
    stale.delete()
    replacement = create_entry(app)
    before = saved_state(replacement)
    with pytest.raises(KeyError):
        apply_operation(stale, operation)
    assert saved_state(replacement) == before


@pytest.mark.parametrize('serializer', ['save', 'reschedule', 'update', 'reservation'])
def test_identity_survives_existing_definition_writers(app, serializer):
    entry = create_entry(app)
    identity = entry.entry_id
    if serializer == 'save':
        entry.save()
    elif serializer == 'reschedule':
        entry.reschedule()
    elif serializer == 'update':
        entry.kwargs = {'payload': 'updated'}
        entry.update_definition()
    else:
        assert next(entry).entry_id == identity
    assert load(entry).entry_id == identity
    assert 'entry_id' not in load(entry).kwargs
    assert 'entry_id' not in json.loads(app.redbeat_redis.hget(entry.key, 'definition'))
    assert json.loads(app.redbeat_redis.hget(entry.key, 'entry_id')) == identity


def test_disabled_identified_entry_still_rejects_stale_reservation(app):
    entry = create_entry(app)
    identity = entry.entry_id
    app.conf.redbeat_entry_policy = lambda following: (None, None)
    next(entry)
    stale = load(entry)
    assert stale.entry_id == identity
    assert stale.jitter_fingerprint is None
    stale.delete()
    replacement = RedBeatSchedulerEntry(
        stale.name, 'tasks.test', schedule(60, app=app), app=app
    ).save_guarded(create=True)
    before = saved_state(replacement)
    with pytest.raises(KeyError):
        next(stale)
    assert saved_state(replacement) == before


def test_policy_adoption_and_updates_do_not_assign_legacy_identity(app):
    entry = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app
    ).save()
    entry.save_guarded()
    app.conf.redbeat_entry_policy = lambda following: (None, FINGERPRINT)
    next(load(entry))
    managed = load(entry)
    managed.kwargs = {'payload': 'updated'}
    managed.update_definition()
    assert load(entry).entry_id is None
    assert load(entry).jitter_fingerprint == FINGERPRINT
    assert 'entry_id' not in json.loads(app.redbeat_redis.hget(entry.key, 'definition'))
    assert not app.redbeat_redis.hexists(entry.key, 'entry_id')


@pytest.mark.parametrize('phased', [False, True])
def test_creation_identity_adds_no_unknown_fields_for_legacy_constructor(app, phased):
    if phased:
        entry = create_entry(app)
    else:
        entry = RedBeatSchedulerEntry(
            'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app
        ).save_guarded(create=True)
    definition = entry.decode_definition(app.redbeat_redis.hget(entry.key, 'definition'))
    metadata = entry.decode_meta(app.redbeat_redis.hget(entry.key, 'meta'))
    assert 'entry_id' not in definition
    assert json.loads(app.redbeat_redis.hget(entry.key, 'entry_id')) == entry.entry_id
    for field in ('enabled', 'run_immediately', 'target'):
        definition.pop(field, None)
    definition.update(metadata)
    legacy = ScheduleEntry(app=app, **definition)
    assert legacy.name == entry.name
    assert legacy.task == entry.task
    assert legacy.last_run_at == entry.last_run_at
    assert legacy.kwargs == entry.kwargs


def test_entry_load_uses_existing_hget_pattern_and_checks_creation_identity(app):
    entry = create_entry(app, kwargs={'payload': 'current'})
    pipeline_class = type(app.redbeat_redis.pipeline())
    original_execute = pipeline_class.execute
    commands = []

    def execute(pipe, *args, **kwargs):
        commands.extend(command for command, _ in pipe.command_stack)
        return original_execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', execute):
        loaded = load(entry)

    assert commands == [
        ('HGET', entry.key, 'entry_id'),
        ('HGET', entry.key, 'definition'),
        ('HGET', entry.key, 'meta'),
        ('HGET', entry.key, 'jitter_fingerprint'),
        ('HGET', entry.key, 'entry_id'),
    ]
    assert loaded.entry_id == entry.entry_id
    assert loaded.kwargs == entry.kwargs
    assert loaded.last_run_at == entry.last_run_at
    assert loaded.jitter_fingerprint == entry.jitter_fingerprint


@pytest.mark.parametrize('value', [None, '', '"creation-id"', b'"creation-id"'])
def test_optional_json_decoder_accepts_missing_text_and_bytes(value):
    assert RedBeatSchedulerEntry.decode_json(value) == ('creation-id' if value else None)


@pytest.mark.parametrize('writer', ['save', 'reschedule'])
@pytest.mark.parametrize('target', [None, 'generic-library-target'])
def test_identified_ordinary_immediate_reservation_keeps_legacy_serializers(app, writer, target):
    entry = RedBeatSchedulerEntry(
        'ordinary-entry',
        'tasks.test',
        schedule(60, app=app),
        app=app,
        kwargs={'payload': 'current'},
        target=target,
        run_immediately=True,
    ).save_guarded(create=True)
    getattr(entry, writer)()
    current = load(entry)
    assert current.is_due()[0]
    assert not current._strtobool(current.run_immediately)
    outgoing = next(current)
    stored = load(entry)
    assert outgoing.kwargs == stored.kwargs == {'payload': 'current'}
    assert stored.entry_id == entry.entry_id
    assert stored.total_run_count == 1
    assert not stored._strtobool(stored.run_immediately)
    assert not app.redbeat_redis.sismember('run_immediately', entry.key)


def test_disabled_identified_policy_can_consume_an_ordinary_immediate_request(app):
    entry = create_entry(app)
    app.conf.redbeat_entry_policy = lambda following: (None, None)
    next(entry)
    current = load(entry)
    assert current.jitter_fingerprint is None
    current.run_immediately = True
    current.target = 'generic-library-target'
    current.reschedule()
    immediate = load(entry)
    previous_count = immediate.total_run_count
    assert immediate.is_due()[0]
    next(immediate)
    stored = load(entry)
    assert stored.total_run_count == previous_count + 1
    assert stored.entry_id == entry.entry_id
    assert stored.jitter_fingerprint is None
    assert not stored._strtobool(stored.run_immediately)


@pytest.mark.parametrize('repeat', [False, True])
def test_nontransactional_load_retries_replacement_without_mixing_generations(app, repeat):
    client = app.redbeat_redis
    entry = create_entry(app, kwargs={'payload': 'old'})
    pipeline_class = type(client.pipeline())
    original_pipeline = client.pipeline
    original_execute = pipeline_class._execute_pipeline
    replacements = []

    def pipeline(*args, **kwargs):
        return original_pipeline(*args, **dict(kwargs, transaction=False))

    def execute(pipe, connection, commands, raise_on_error):
        commands = list(commands)
        if not commands or commands[0][0] != ('HGET', entry.key, 'entry_id'):
            return original_execute(pipe, connection, commands, raise_on_error)
        results = []
        for index, (command, options) in enumerate(commands):
            connection.send_command(*command)
            results.append(pipe.parse_response(connection, command[0], **options))
            if index == 1 and (repeat or not replacements):
                entry.delete()
                replacements.append(create_entry(app, kwargs={'payload': 'replacement'}))
        return results

    with patch.object(client, 'pipeline', pipeline), patch.object(
        pipeline_class, '_execute_pipeline', execute
    ):
        if repeat:
            with pytest.raises(WatchError):
                load(entry)
        else:
            loaded = load(entry)
            assert loaded.entry_id == replacements[-1].entry_id
            assert loaded.kwargs == {'payload': 'replacement'}

    assert len(replacements) == (3 if repeat else 1)
    assert load(entry).entry_id == replacements[-1].entry_id


def test_exhausted_load_conflict_defers_tick_without_publication_or_index_removal(app):
    entry = create_entry(app)
    scheduler = RedBeatScheduler(app=app, lazy=True, lock_key=None)
    scheduler.lock_key = None
    before = saved_state(entry)
    with patch.object(RedBeatSchedulerEntry, 'from_key', side_effect=WatchError), patch.object(
        scheduler, 'apply_async'
    ) as publish:
        assert scheduler.tick() == min(1, scheduler.max_interval)
    publish.assert_not_called()
    assert saved_state(entry) == before
