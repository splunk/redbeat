import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from queue import Empty
from uuid import uuid4

import pytest
from celery import Celery
from celery.schedules import schedule
from redis import Redis
from redis.exceptions import ConnectionError

from redbeat.decoder import to_timestamp
from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
FINGERPRINT = 'test-policy-v1'


@pytest.fixture
def app(tmp_path):
    executable = os.environ.get('REDBEAT_TEST_REDIS_SERVER')
    if not executable:
        pytest.skip('Set REDBEAT_TEST_REDIS_SERVER to an isolated Redis test binary')
    executable = shutil.which(executable)
    if not executable:
        pytest.fail('The requested Redis test binary is unavailable')
    socket_path = str(tmp_path / 'redis.sock')
    process = subprocess.Popen(
        [
            executable,
            '--port',
            '0',
            '--unixsocket',
            socket_path,
            '--unixsocketperm',
            '700',
            '--save',
            '',
            '--appendonly',
            'no',
            '--dir',
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    client = Redis(unix_socket_path=socket_path, decode_responses=True)
    result = None
    try:
        deadline = time.monotonic() + 5
        while True:
            if process.poll() is not None:
                pytest.fail('Isolated Redis exited before startup')
            try:
                client.ping()
                break
            except ConnectionError:
                if time.monotonic() >= deadline:
                    pytest.fail('Isolated Redis did not become ready')
                time.sleep(0.02)
        result = Celery('real-jitter-tests', broker='memory://', set_as_current=False)
        result.conf.update(
            redbeat_redis_url='redis://unused/0',
            timezone='UTC',
            task_default_queue='jitter-' + uuid4().hex,
        )
        result.redbeat_redis = client
        yield result
    finally:
        if result is not None:
            result.close()
        client.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def read_message(queue):
    try:
        return queue.get(block=False)
    except Empty:
        return None


@pytest.mark.parametrize('interval,first_delay', [(60, 102), (300, 356), (600, 22), (1800, 35)])
def test_real_redis_reload_lookahead_and_beat_publication(app, interval, first_delay):
    app.now = lambda: NOW
    due = NOW + timedelta(seconds=first_delay)
    initial = RedBeatSchedulerEntry(
        name='test-first-run',
        task='tasks.test',
        schedule=schedule(interval, app=app),
        kwargs={'address': 'router-01:161'},
        options={'expires': 120},
        app=app,
    )
    initial.save_with_first_due(due, FINGERPRINT, first_run_kwargs={'bootstrap_token': 'first'})
    loaded = RedBeatSchedulerEntry.from_key(initial.key, app)
    assert loaded.last_run_at == due - timedelta(seconds=interval)
    assert loaded.jitter_fingerprint == FINGERPRINT
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == to_timestamp(due)
    # An ordinary payload update must leave the first due unchanged.
    loaded.kwargs['profile'] = 'updated'
    loaded.update_definition()
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=3600)
    with app.connection_for_read() as connection:
        queue = connection.SimpleQueue(app.conf.task_default_queue)
        try:
            for now in (NOW, due - timedelta(microseconds=1)):
                app.now = lambda: now
                assert 'test-first-run' in scheduler.schedule  # Real lookahead.
                scheduler.tick()
                assert read_message(queue) is None
            app.now = lambda: due
            # A replacement scheduler models Beat reload with Redis retained.
            scheduler = RedBeatScheduler(app, lazy=True, max_interval=3600)
            scheduler.tick()
            message = read_message(queue)
            assert message is not None
            assert message.payload[1]['bootstrap_token'] == 'first'
            assert message.payload[1]['profile'] == 'updated'
            assert message.headers.get('expires') is not None
            message.ack()
            assert read_message(queue) is None
            stored = RedBeatSchedulerEntry.from_key(initial.key, app)
            assert 'bootstrap_token' not in stored.kwargs
            assert stored.first_run_kwargs == {}
            assert stored.total_run_count == 1
            assert stored.last_run_at == due
            next_due = due + timedelta(seconds=interval)
            assert app.redbeat_redis.zscore(
                app.redbeat_conf.schedule_key, initial.key
            ) == to_timestamp(next_due)
            app.now = lambda: next_due - timedelta(microseconds=1)
            scheduler.tick()
            assert read_message(queue) is None
            app.now = lambda: next_due
            scheduler.tick()
            message = read_message(queue)
            assert message is not None
            assert 'bootstrap_token' not in message.payload[1]
            message.ack()
            assert list(app.redbeat_redis.scan_iter('redbeat:test-first-run')) == [initial.key]
        finally:
            queue.close()
            scheduler.close()


def test_real_clock_dispatch_is_not_early_and_recurs(app):
    due = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=2)
    initial = RedBeatSchedulerEntry(
        'real-clock-test',
        'tasks.test',
        schedule(2, app=app),
        app=app,
    ).save_with_first_due(due, FINGERPRINT)
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=0.05)
    observed = []
    with app.connection_for_read() as connection:
        queue = connection.SimpleQueue(app.conf.task_default_queue)
        try:
            deadline = time.monotonic() + 8
            while len(observed) < 2 and time.monotonic() < deadline:
                scheduler.tick()
                message = read_message(queue)
                if message is not None:
                    observed.append(datetime.now(timezone.utc))
                    message.ack()
                time.sleep(0.02)
            assert len(observed) == 2
            assert timedelta(0) <= observed[0] - due < timedelta(seconds=1)
            # Ordinary interval starts at reservation, not at the old hash phase.
            assert 1.8 <= (observed[1] - observed[0]).total_seconds() < 3
            assert RedBeatSchedulerEntry.from_key(initial.key, app).total_run_count == 2
        finally:
            queue.close()
            scheduler.close()
