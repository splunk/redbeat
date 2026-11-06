import json
from datetime import timedelta

import pytest
from celery.beat import ScheduleEntry
from celery.schedules import schedule

from redbeat.decoder import RedBeatJSONDecoder, RedBeatJSONEncoder
from redbeat.schedulers import RedBeatSchedulerEntry
from tests.test_jitter_concurrent_updates import app as parity_app
from tests.test_jitter_integration import NOW
from tests.test_jitter_integration import app as redis_app  # noqa: F401

app = parity_app


def original_write(entry, operation):
    if operation in ('reserve', 'reserve_without_count'):
        outgoing = ScheduleEntry._next_instance(entry, last_run_at=entry.app.now())
        if operation == 'reserve_without_count':
            outgoing.total_run_count = entry.total_run_count
        metadata = {
            'last_run_at': outgoing.last_run_at,
            'total_run_count': outgoing.total_run_count,
        }
        with entry.app.redbeat_redis.pipeline() as pipe:
            pipe.hset(entry.key, 'meta', json.dumps(metadata, cls=RedBeatJSONEncoder))
            pipe.zadd(entry.app.redbeat_conf.schedule_key, {entry.key: outgoing.score})
            pipe.execute()
        return outgoing

    reset = operation == 'reschedule'
    if reset:
        entry.last_run_at = entry.app.now()
    definition = {
        'name': entry.name,
        'task': entry.task,
        'args': entry.args,
        'kwargs': entry.kwargs,
        'options': entry.options,
        'schedule': entry.schedule,
        'enabled': entry.enabled,
        'run_immediately': entry.run_immediately,
    }
    if reset:
        definition['target'] = entry.target
    if entry.first_run_kwargs:
        definition['first_run_kwargs'] = dict(entry.first_run_kwargs)
    metadata = {'last_run_at': entry.last_run_at}
    with entry.app.redbeat_redis.pipeline() as pipe:
        pipe.hset(entry.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
        if reset:
            pipe.hset(entry.key, 'meta', json.dumps(metadata, cls=RedBeatJSONEncoder))
        else:
            pipe.hsetnx(entry.key, 'meta', json.dumps(metadata, cls=RedBeatJSONEncoder))
        pipe.zadd(entry.app.redbeat_conf.schedule_key, {entry.key: entry.score})
        if entry._strtobool(entry.run_immediately):
            pipe.sadd('run_immediately', entry.key)
        pipe.execute()
    return None


def prepared_entry(app, enabled, immediate):
    entry = RedBeatSchedulerEntry(
        'persistence-parity',
        'tasks.test',
        schedule(210, app=app),
        app=app,
        last_run_at=NOW - timedelta(seconds=210),
        kwargs={'value': 'old'},
    )
    client = app.redbeat_redis
    client.delete(entry.key)
    client.zrem(app.redbeat_conf.schedule_key, entry.key)
    client.srem('run_immediately', entry.key)
    original_write(entry, 'save')
    client.hset(
        entry.key,
        mapping={
            'meta': json.dumps(
                {'last_run_at': entry.last_run_at, 'total_run_count': 7},
                cls=RedBeatJSONEncoder,
            ),
            'owner_note': 'keep this independent field',
        },
    )
    if not immediate:
        client.sadd('run_immediately', entry.key)
    entry = RedBeatSchedulerEntry.from_key(entry.key, app=app)
    entry.args = ['changed-argument']
    entry.kwargs = {'value': 'new', 'nested': {'revision': 2}}
    entry.options = {'queue': 'parity-queue', 'expires': 120}
    entry.target = 'parity-owner'
    entry.enabled = str(enabled)
    entry.run_immediately = str(immediate)
    entry.first_run_kwargs = {'bootstrap': 'first-invocation'}
    return entry


def persisted_state(entry, outgoing):
    client = entry.app.redbeat_redis
    fields = client.hgetall(entry.key)
    for field in ('definition', 'meta'):
        if field in fields:
            fields[field] = json.loads(fields[field], cls=RedBeatJSONDecoder)
    invocation = None
    if outgoing is not None:
        invocation = {
            field: getattr(outgoing, field)
            for field in (
                'name',
                'task',
                'args',
                'kwargs',
                'options',
                'schedule',
                'target',
                'enabled',
                'run_immediately',
                'first_run_kwargs',
                'last_run_at',
                'total_run_count',
            )
        }
    return (
        fields,
        client.zscore(entry.app.redbeat_conf.schedule_key, entry.key),
        client.sismember('run_immediately', entry.key),
        invocation,
    )


@pytest.mark.parametrize(
    'operation',
    ['save', 'reschedule', 'guarded_save', 'reserve', 'reserve_without_count'],
)
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('immediate', [False, True])
@pytest.mark.parametrize('microseconds', [0, 654321])
def test_ordinary_persistence_matches_original_patterns(
    app, operation, enabled, immediate, microseconds
):
    app.now = lambda: NOW + timedelta(seconds=5, microseconds=microseconds)
    reference = prepared_entry(app, enabled, immediate)
    expected = persisted_state(reference, original_write(reference, operation))

    current = prepared_entry(app, enabled, immediate)
    outgoing = None
    if operation == 'save':
        current.save()
    elif operation == 'reschedule':
        current.reschedule()
    elif operation == 'guarded_save':
        current.save_guarded()
    else:
        outgoing = current.next(
            last_run_at=app.now(), only_update_last_run_at=operation == 'reserve_without_count'
        )

    actual = persisted_state(current, outgoing)
    assert actual == expected
    assert 'jitter_fingerprint' not in actual[0]
    assert actual[0]['owner_note'] == 'keep this independent field'
