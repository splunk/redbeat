import json
from datetime import timedelta
from unittest.mock import patch

import pytest

from redbeat.decoder import RedBeatJSONEncoder
from tests.test_jitter_concurrent_updates import app as efficiency_app
from tests.test_jitter_concurrent_updates import create_entry, load
from tests.test_jitter_integration import NOW
from tests.test_jitter_integration import app as redis_app  # noqa: F401

app = efficiency_app


@pytest.mark.parametrize('run_immediately', [False, True])
@pytest.mark.parametrize('changed_policy', [False, True])
def test_request_writes_fingerprint_only_when_the_applied_policy_changes(
    app, run_immediately, changed_policy
):
    entry = create_entry(app)
    fingerprint = 'new-policy' if changed_policy else entry.jitter_fingerprint
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    commands = []

    def record(pipe, *args, **kwargs):
        commands.extend(command for command, _ in pipe.command_stack)
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', record):
        entry.request_run(
            NOW if run_immediately else NOW + timedelta(seconds=20),
            run_immediately=run_immediately,
            fingerprint=fingerprint,
        )

    fingerprint_writes = [
        command for command in commands if command[:3] == ('HSET', entry.key, 'jitter_fingerprint')
    ]
    assert len(fingerprint_writes) == int(changed_policy)
    stored = load(entry)
    assert stored.jitter_fingerprint == fingerprint
    assert stored._strtobool(stored.run_immediately)
    due = NOW if run_immediately else NOW + timedelta(seconds=20)
    assert stored.last_run_at == due - timedelta(seconds=60)


@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize(
    'kwargs',
    [{}, {'payload': {'__type__': 'custom', 'value': 'unchanged'}}],
    ids=['ordinary', 'custom-type'],
)
def test_identical_definition_keeps_pending_state_without_writing(app, pending, kwargs):
    entry = create_entry(app, first_run_kwargs={'bootstrap': 'first'}, kwargs=kwargs)
    if pending:
        entry.request_run(NOW + timedelta(seconds=20))
    before = app.redbeat_redis.hgetall(entry.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
    pipeline_class = type(app.redbeat_redis.pipeline())

    with patch.object(pipeline_class, 'execute') as execute:
        entry.update_definition()

    execute.assert_not_called()
    assert app.redbeat_redis.hgetall(entry.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == score


def test_changed_definition_still_writes_and_preserves_pending_request(app):
    entry = create_entry(app, kwargs={'value': 'old'})
    entry.request_run(NOW + timedelta(seconds=20))
    before = app.redbeat_redis.hgetall(entry.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
    entry.kwargs = {'value': 'new'}
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    commands = []

    def record(pipe, *args, **kwargs):
        commands.extend(command for command, _ in pipe.command_stack)
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', record):
        entry.update_definition()

    assert len(commands) == 1
    assert commands[0][:3] == ('HSET', entry.key, 'definition')
    after = app.redbeat_redis.hgetall(entry.key)
    assert after['definition'] != before['definition']
    assert {field: value for field, value in after.items() if field != 'definition'} == {
        field: value for field, value in before.items() if field != 'definition'
    }
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == score
    assert load(entry).kwargs == {'value': 'new'}


@pytest.mark.parametrize('field', ['args', 'kwargs'])
@pytest.mark.parametrize('previous,updated', [(True, 1), (1, 1.0)])
def test_definition_update_preserves_argument_type_changes(app, field, previous, updated):
    original_value = [previous] if field == 'args' else {'value': previous}
    entry = create_entry(app, **{field: original_value})
    setattr(entry, field, [updated] if field == 'args' else {'value': updated})

    entry.update_definition()

    stored_value = getattr(load(entry), field)
    value = stored_value[0] if field == 'args' else stored_value['value']
    assert value == updated
    assert type(value) is type(updated)


@pytest.mark.parametrize('changed_state', ['policy', 'missing'])
def test_identical_definition_still_rejects_incompatible_stored_state(app, changed_state):
    entry = create_entry(app)
    if changed_state == 'missing':
        entry.delete()
        error = KeyError
    else:
        app.redbeat_redis.hset(entry.key, 'jitter_fingerprint', json.dumps('changed'))
        error = ValueError
    before = app.redbeat_redis.hgetall(entry.key)
    pipeline_class = type(app.redbeat_redis.pipeline())

    with patch.object(pipeline_class, 'execute') as execute:
        with pytest.raises(error):
            entry.update_definition()

    execute.assert_not_called()
    assert app.redbeat_redis.hgetall(entry.key) == before


def test_identical_definition_rejects_identical_timing_same_key_replacement(app):
    stale = create_entry(app)
    stale.delete()
    replacement = create_entry(app)
    before = app.redbeat_redis.hgetall(replacement.key)
    pipeline_class = type(app.redbeat_redis.pipeline())

    with patch.object(pipeline_class, 'execute') as execute:
        with pytest.raises(KeyError):
            stale.update_definition()

    execute.assert_not_called()
    assert app.redbeat_redis.hgetall(replacement.key) == before


def test_noop_definition_does_not_replace_a_concurrent_payload_update(app):
    entry = create_entry(app, kwargs={'value': 'old'})
    pipeline_class = type(app.redbeat_redis.pipeline())
    hget = pipeline_class.hget
    changed = []

    def update_after_read(pipe, *args, **kwargs):
        value = hget(pipe, *args, **kwargs)
        if not changed and args == (entry.key, 'definition'):
            definition = entry.decode_definition(value)
            definition['kwargs'] = {'value': 'new'}
            app.redbeat_redis.hset(
                entry.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder)
            )
            changed.append(True)
        return value

    with (
        patch.object(pipeline_class, 'hget', update_after_read),
        patch.object(pipeline_class, 'execute') as execute,
    ):
        entry.update_definition()

    execute.assert_not_called()
    assert changed == [True]
    assert load(entry).kwargs == {'value': 'new'}
