import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
import redis
from celery import Celery
from celery.schedules import schedule
from redis import Redis
from redis.exceptions import ConnectionError

from redbeat.decoder import RedBeatJSONEncoder, to_timestamp
from redbeat.schedulers import (
    EntryAlreadyExists,
    RedBeatScheduler,
    RedBeatSchedulerEntry,
    RetryingConnection,
)

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
POLICY = 'library-policy-one'


@dataclass(repr=False)
class ClusterNodes:
    clients: list
    ports: list
    password: str


def reserve_node_ports():
    while True:
        command_socket = socket.socket()
        command_socket.bind(('127.0.0.1', 0))
        port = command_socket.getsockname()[1]
        if port > 55535:
            command_socket.close()
            continue
        bus_socket = socket.socket()
        try:
            bus_socket.bind(('127.0.0.1', port + 10000))
        except OSError:
            command_socket.close()
            bus_socket.close()
            continue
        return port, (command_socket, bus_socket)


@pytest.fixture(scope='module')
def cluster_nodes(tmp_path_factory):
    executable = os.environ.get('REDBEAT_TEST_REDIS_SERVER')
    if not executable:
        pytest.skip('Set REDBEAT_TEST_REDIS_SERVER to an isolated Redis test binary')
    if tuple(int(part) for part in redis.__version__.split('.')[:2]) < (6, 1):
        pytest.skip('Managed Cluster transactions require redis-py 6.1 or later')
    executable = shutil.which(executable)
    if not executable:
        pytest.fail('The requested Redis test binary is unavailable')
    directory = tmp_path_factory.mktemp('managed-cluster')
    password = uuid4().hex
    processes = []
    clients = []
    ports = []
    try:
        for index in range(3):
            node_directory = directory / str(index)
            node_directory.mkdir()
            port, reservations = reserve_node_ports()
            for reservation in reservations:
                reservation.close()
            process = subprocess.Popen(
                [executable, '-'],
                cwd=node_directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)
            process.stdin.write(
                '\n'.join(
                    [
                        'bind 127.0.0.1',
                        f'port {port}',
                        'protected-mode yes',
                        f'requirepass {password}',
                        f'masterauth {password}',
                        'cluster-enabled yes',
                        'cluster-config-file nodes.conf',
                        'cluster-node-timeout 1000',
                        'cluster-announce-ip 127.0.0.1',
                        f'cluster-announce-port {port}',
                        f'cluster-announce-bus-port {port + 10000}',
                        'save ""',
                        'appendonly no',
                        f'dir {json.dumps(str(node_directory))}',
                        '',
                    ]
                )
            )
            process.stdin.close()
            client = Redis(
                host='127.0.0.1',
                port=port,
                password=password,
                socket_timeout=1,
                decode_responses=True,
            )
            clients.append(client)
            ports.append(port)
            deadline = time.monotonic() + 5
            while True:
                if process.poll() is not None:
                    pytest.fail('An isolated Cluster node exited before startup')
                try:
                    client.ping()
                    break
                except ConnectionError:
                    if time.monotonic() >= deadline:
                        pytest.fail('An isolated Cluster node did not become ready')
                    time.sleep(0.02)
            client.cluster('SET-CONFIG-EPOCH', index + 1)
            start = index * 16384 // 3
            end = (index + 1) * 16384 // 3
            client.cluster('ADDSLOTS', *range(start, end))
        for port in ports[1:]:
            clients[0].cluster('MEET', '127.0.0.1', port)
        deadline = time.monotonic() + 10
        while not all(client.cluster('INFO')['cluster_state'] == 'ok' for client in clients):
            if time.monotonic() >= deadline:
                pytest.fail('The isolated Cluster did not cover all slots')
            time.sleep(0.05)
        yield ClusterNodes(clients, ports, password)
    finally:
        for client in clients:
            client.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture(params=[True, False], ids=['text', 'bytes'])
def cluster_app(cluster_nodes, request):
    from redis.cluster import ClusterNode, RedisCluster

    nodes = cluster_nodes.clients
    ports = cluster_nodes.ports
    password = cluster_nodes.password
    client = RedisCluster(
        startup_nodes=[ClusterNode('127.0.0.1', port) for port in ports],
        password=password,
        decode_responses=request.param,
        socket_timeout=2,
    )
    app = Celery('managed-cluster-tests', broker='memory://', set_as_current=False)
    app.conf.update(
        redbeat_redis_url='redis://unused/0',
        redbeat_key_prefix='{run_immediately}:tests:' + uuid4().hex + ':',
        task_default_queue='managed-cluster-' + uuid4().hex,
        timezone='UTC',
    )
    app.redbeat_redis = client
    app.now = lambda: NOW
    try:
        yield app
    finally:
        app.close()
        client.close()
        for node in nodes:
            node.flushdb()


def create_entry(app, due=NOW, **kwargs):
    return RedBeatSchedulerEntry(
        'managed-entry', 'tasks.example', schedule(60, app=app), app=app, **kwargs
    ).save_with_first_due(due, POLICY)


def load(entry):
    return RedBeatSchedulerEntry.from_key(entry.key, app=entry.app)


def stored_state(entry):
    client = entry.app.redbeat_redis
    return (
        client.hgetall(entry.key),
        client.zscore(entry.app.redbeat_conf.schedule_key, entry.key),
        client.sismember('run_immediately', entry.key),
    )


@pytest.mark.parametrize('writer', ['save', 'reschedule'])
def test_cluster_legacy_immediate_write_replaces_managed_timing(cluster_app, writer):
    entry = create_entry(cluster_app, NOW + timedelta(seconds=20))
    identity = entry.entry_id
    entry.run_immediately = True
    getattr(entry, writer)()
    stored = load(entry)
    assert stored.jitter_fingerprint is None
    assert stored.entry_id == identity
    assert stored.is_due()[0]
    next(stored)
    assert load(entry).last_run_at == NOW
    assert load(entry).total_run_count == 1


def test_cluster_managed_creation_payload_refresh_and_reservation(cluster_app):
    app = cluster_app
    due = NOW + timedelta(seconds=20)
    entry = create_entry(
        app,
        due,
        kwargs={'value': 'original'},
        first_run_kwargs={'initial_only': True},
    )
    loaded = load(entry)
    assert loaded.jitter_fingerprint == POLICY
    assert json.loads(app.redbeat_redis.hget(entry.key, 'jitter_fingerprint')) == POLICY
    assert loaded.last_run_at == due - timedelta(seconds=60)
    before = stored_state(loaded)
    loaded.kwargs['value'] = 'updated'
    loaded.update_definition()
    assert stored_state(loaded)[1:] == before[1:]
    app.now = lambda: due
    outgoing = next(load(entry))
    assert outgoing.kwargs == {'value': 'updated', 'initial_only': True}
    saved = load(entry)
    assert saved.kwargs == {'value': 'updated'}
    assert saved.first_run_kwargs == {}
    assert saved.total_run_count == 1
    assert stored_state(saved)[1] == to_timestamp(due + timedelta(seconds=60))

    requested = due + timedelta(seconds=15)
    saved.request_run(requested)
    assert not app.redbeat_redis.sismember('run_immediately', entry.key)
    load(entry).request_run(requested + timedelta(seconds=10))
    assert stored_state(entry)[1] == to_timestamp(requested)
    app.now = lambda: due + timedelta(seconds=1)
    load(entry).request_run(app.now(), run_immediately=True)
    assert app.redbeat_redis.sismember('run_immediately', entry.key)
    urgent = load(entry)
    assert urgent.is_due()[0]
    next(urgent)
    assert load(entry).total_run_count == 2
    assert not app.redbeat_redis.sismember('run_immediately', entry.key)
    assert stored_state(entry)[1] == to_timestamp(app.now() + timedelta(seconds=60))
    entry.delete()
    assert stored_state(entry) == ({}, None, False)


def test_cluster_guarded_ordinary_update(cluster_app):
    entry = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.example', schedule(60, app=cluster_app), app=cluster_app
    ).save()
    entry.kwargs = {'updated': True}
    updated = entry.save_guarded()
    assert updated.kwargs == {'updated': True}
    assert updated.jitter_fingerprint is None


@pytest.mark.parametrize('phased', [False, True])
def test_cluster_creation_conflict_preserves_winner_and_unset_caller_identity(cluster_app, phased):
    app = cluster_app
    entry = RedBeatSchedulerEntry('creation-race', 'tasks.example', schedule(60, app=app), app=app)
    pipeline_class = type(app.redbeat_redis.pipeline(transaction=True))
    execute = pipeline_class.execute
    winner = []
    winner_state = []

    def commit(pipe, *args, **kwargs):
        if not winner:
            replacement = RedBeatSchedulerEntry(
                entry.name, entry.task, entry.schedule, app=app, kwargs={'payload': 'winner'}
            )
            winner.append(replacement)
            replacement.save_guarded(create=True)
            winner_state.append(stored_state(replacement))
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', commit):
        with pytest.raises(EntryAlreadyExists):
            if phased:
                entry.save_with_first_due(NOW, POLICY)
            else:
                entry.save_guarded(create=True)
    assert entry.entry_id is None
    assert stored_state(entry) == winner_state[0]
    assert load(entry).entry_id == winner[0].entry_id


@pytest.mark.parametrize('operation', ['definition', 'request', 'reservation', 'ordinary'])
def test_cluster_creation_identity_rejects_same_key_replacement(cluster_app, operation):
    app = cluster_app
    if operation == 'ordinary':
        stale = RedBeatSchedulerEntry(
            'managed-entry', 'tasks.example', schedule(60, app=app), app=app
        ).save_guarded(create=True)
    else:
        stale = create_entry(app)
    stale.delete()
    if operation == 'ordinary':
        replacement = RedBeatSchedulerEntry(
            stale.name, 'tasks.example', schedule(60, app=app), app=app
        ).save_guarded(create=True)
    else:
        replacement = create_entry(app)
    assert load(replacement).entry_id == replacement.entry_id
    assert stale.entry_id != replacement.entry_id
    before = stored_state(replacement)
    with pytest.raises(KeyError):
        if operation == 'definition':
            stale.update_definition()
        elif operation == 'request':
            stale.request_run(NOW)
        elif operation == 'ordinary':
            stale.save_guarded()
        else:
            next(stale)
    assert stored_state(replacement) == before


def test_cluster_default_prefix_ordinary_path_remains_usable(cluster_app):
    app = cluster_app
    app.conf.redbeat_key_prefix = 'redbeat:'
    entry = RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.example', schedule(60, app=app), app=app
    ).save()
    assert load(entry).jitter_fingerprint is None
    next(load(entry))
    assert load(entry).total_run_count == 1
    entry.reschedule(NOW + timedelta(seconds=1))
    assert load(entry).last_run_at == NOW + timedelta(seconds=1)
    entry.delete()
    assert stored_state(entry) == ({}, None, False)


@pytest.mark.parametrize('operation', ['create', 'ordinary-update', 'request', 'reserve'])
def test_cluster_cross_slot_timing_write_leaves_state_unchanged(cluster_app, operation):
    from redis.exceptions import RedisClusterException

    app = cluster_app
    app.conf.redbeat_key_prefix = 'redbeat:'
    entry = RedBeatSchedulerEntry(
        'cross-slot-entry', 'tasks.example', schedule(60, app=app), app=app
    )
    if operation != 'create':
        entry.save()
        if operation != 'ordinary-update':
            app.redbeat_redis.hset(entry.key, 'jitter_fingerprint', json.dumps(POLICY))
            entry = load(entry)
    before = stored_state(entry)
    with pytest.raises(RedisClusterException):
        if operation == 'create':
            entry.save_with_first_due(NOW + timedelta(seconds=20), POLICY)
        elif operation == 'ordinary-update':
            entry.kwargs = {'new': True}
            entry.save_guarded()
        elif operation == 'request':
            entry.request_run(NOW + timedelta(seconds=20))
        else:
            next(entry)
    assert stored_state(entry) == before


def test_cluster_single_hash_definition_update_needs_no_shared_tag(cluster_app):
    app = cluster_app
    app.conf.redbeat_key_prefix = 'redbeat:'
    entry = RedBeatSchedulerEntry(
        'single-hash-entry', 'tasks.example', schedule(60, app=app), app=app
    ).save()
    app.redbeat_redis.hset(entry.key, 'jitter_fingerprint', json.dumps(POLICY))
    entry = load(entry)
    before = stored_state(entry)
    entry.kwargs = {'updated': True}
    entry.update_definition()
    assert load(entry).kwargs == {'updated': True}
    assert stored_state(entry)[1:] == before[1:]
    pipeline_class = type(app.redbeat_redis.pipeline(transaction=True))
    with patch.object(pipeline_class, 'execute') as execute:
        entry.update_definition()
    execute.assert_not_called()


def test_cluster_definition_update_retries_a_real_watch_conflict(cluster_app):
    app = cluster_app
    entry = create_entry(app)
    entry.kwargs = {'updated': True}
    pipeline_class = type(app.redbeat_redis.pipeline(transaction=True))
    original_execute = pipeline_class.execute
    attempts = []

    def conflict_once(pipe, *args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            app.redbeat_redis.hset(
                entry.key,
                'meta',
                json.dumps({'last_run_at': NOW, 'total_run_count': 7}, cls=RedBeatJSONEncoder),
            )
        return original_execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict_once):
        entry.update_definition()
    assert len(attempts) == 2
    assert load(entry).kwargs == {'updated': True}
    assert load(entry).total_run_count == 7


def test_cluster_reservation_retries_without_replacing_new_payload(cluster_app):
    app = cluster_app
    entry = create_entry(app, kwargs={'value': 'original'})
    pipeline_class = type(app.redbeat_redis.pipeline(transaction=True))
    original_execute = pipeline_class.execute
    attempts = []

    def conflict_once(pipe, *args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            definition = json.loads(app.redbeat_redis.hget(entry.key, 'definition'))
            definition['kwargs'] = {'value': 'newer'}
            app.redbeat_redis.hset(
                entry.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder)
            )
        return original_execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict_once):
        outgoing = next(entry)
    assert len(attempts) == 2
    assert outgoing.kwargs == {'value': 'original'}
    assert load(entry).kwargs == {'value': 'newer'}
    assert load(entry).total_run_count == 1


def test_cluster_reservation_rejects_a_newer_run(cluster_app):
    entry = create_entry(cluster_app)
    stale = load(entry)
    next(entry)
    before = stored_state(entry)
    with pytest.raises(ValueError, match='run state'):
        next(stale)
    assert stored_state(entry) == before


def test_cluster_generic_policy_changes_only_the_following_run(cluster_app):
    app = cluster_app
    due = NOW + timedelta(seconds=20)
    entry = create_entry(app, due)
    calls = []

    def policy(following):
        calls.append(following.jitter_fingerprint)
        if following.jitter_fingerprint == POLICY:
            return following.last_run_at + timedelta(seconds=80), 'library-policy-two'
        return None

    app.conf.redbeat_entry_policy = policy
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=3600)
    try:
        with app.connection_for_read() as connection:
            queue = connection.SimpleQueue(app.conf.task_default_queue)
            try:
                app.now = lambda: due - timedelta(seconds=1)
                scheduler.tick()
                assert calls == []
                assert queue.qsize() == 0
                app.now = lambda: due
                scheduler.tick()
                message = queue.get(block=False)
                message.ack()
                assert queue.qsize() == 0
                assert calls == [POLICY]
                assert load(entry).jitter_fingerprint == 'library-policy-two'
                following_due = due + timedelta(seconds=80)
                assert stored_state(entry)[1] == to_timestamp(following_due)
                app.now = lambda: following_due
                scheduler.tick()
                message = queue.get(block=False)
                message.ack()
                assert load(entry).total_run_count == 2
                assert stored_state(entry)[1] == to_timestamp(following_due + timedelta(seconds=60))
            finally:
                queue.close()
    finally:
        scheduler.close()


def test_cluster_retrying_connection_preserves_transaction_behavior(cluster_app):
    app = cluster_app
    app.redbeat_redis = RetryingConnection(0, app.redbeat_redis)
    entry = create_entry(app)
    loaded = load(entry)
    loaded.kwargs = {'updated': True}
    loaded.update_definition()
    loaded.request_run(NOW + timedelta(seconds=10))
    app.now = lambda: NOW + timedelta(seconds=10)
    next(load(entry))
    assert load(entry).total_run_count == 1
    assert load(entry).kwargs == {'updated': True}


def test_cluster_fractional_deadlines_preserve_precision_and_request_state(cluster_app):
    app = cluster_app
    local_zone = timezone(timedelta(hours=5.5))
    due = NOW + timedelta(microseconds=250000)
    entry = RedBeatSchedulerEntry(
        'fractional-timing', 'tasks.example', schedule(1.5, app=app), app=app
    ).save_with_first_due(due.astimezone(local_zone), POLICY)
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=60)
    try:
        with patch.object(app, 'send_task') as publish:
            for count, deadline in enumerate((due, due + timedelta(seconds=0.8)), 1):
                app.now = lambda: deadline - timedelta(microseconds=1)
                scheduler.tick()
                publish.assert_not_called()
                app.now = lambda: deadline
                scheduler.tick()
                publish.assert_called_once()
                publish.reset_mock()
                assert load(entry).total_run_count == count
                if count == 1:
                    load(entry).request_run((due + timedelta(seconds=0.8)).astimezone(local_zone))
                    following_due = due + timedelta(seconds=3.3)
                    app.conf.redbeat_entry_policy = lambda following: (
                        following_due.astimezone(local_zone),
                        'library-policy-two',
                    )
        stored = load(entry)
        assert stored.jitter_fingerprint == 'library-policy-two'
        assert stored.last_run_at == following_due - timedelta(seconds=1.5)
        assert stored.last_run_at.tzinfo == timezone.utc
        assert stored_state(entry)[1] == to_timestamp(following_due)
    finally:
        scheduler.close()


def test_cluster_json_fingerprint_roundtrips_with_non_default_encoding(cluster_app, cluster_nodes):
    from redis.cluster import ClusterNode, RedisCluster

    app = cluster_app
    ports = cluster_nodes.ports
    password = cluster_nodes.password
    original = app.redbeat_redis
    client = RedisCluster(
        startup_nodes=[ClusterNode('127.0.0.1', port) for port in ports],
        password=password,
        encoding='latin-1',
        decode_responses=original.get_encoder().decode_responses,
        socket_timeout=2,
    )
    app.redbeat_redis = client
    try:
        fingerprint = 'library-policy-caf\u00e9-\u03bb'
        entry = RedBeatSchedulerEntry(
            'encoded-marker', 'tasks.example', schedule(60, app=app), app=app
        ).save_with_first_due(NOW, fingerprint)
        raw = client.hget(entry.key, 'jitter_fingerprint')
        assert json.loads(raw) == fingerprint
        current = load(entry)
        assert current.jitter_fingerprint == fingerprint
        current.kwargs = {'updated': True}
        current.update_definition()
        current.request_run(NOW + timedelta(seconds=10))
        app.now = lambda: NOW + timedelta(seconds=10)
        next(load(entry))
        assert load(entry).jitter_fingerprint == fingerprint
        assert load(entry).total_run_count == 1
    finally:
        app.redbeat_redis = original
        client.close()
