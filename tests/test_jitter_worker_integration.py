import time
from datetime import datetime, timedelta, timezone
from threading import Event
from uuid import uuid4

import pytest
from celery.contrib.testing.worker import start_worker
from celery.schedules import schedule

from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_integration import FINGERPRINT
from tests.test_jitter_integration import app as redis_app

app = redis_app


@pytest.fixture
def broker_app(app):
    path = app.redbeat_redis.connection_pool.connection_kwargs['path']
    app.conf.update(
        broker_url='redis+socket://' + path + '?virtual_host=1',
        result_backend=None,
        task_ignore_result=True,
        worker_hijack_root_logger=False,
        worker_enable_remote_control=False,
        worker_prefetch_multiplier=1,
    )
    return app


def worker(app):
    return start_worker(
        app,
        pool='solo',
        concurrency=1,
        perform_ping_check=False,
        without_heartbeat=True,
        loglevel='ERROR',
        shutdown_timeout=10,
    )


def save_entry(app, task, interval=600, options=None):
    due = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=2)
    entry = RedBeatSchedulerEntry(
        'worker-' + uuid4().hex,
        task,
        schedule(interval, app=app),
        options=options or {},
        app=app,
    ).save_with_first_due(due, FINGERPRINT, first_run_kwargs={'bootstrap_token': 'first'})
    return entry, due


def test_real_broker_worker_first_context_and_recurrence(broker_app):
    app = broker_app
    calls = []
    completed = Event()

    @app.task(name='validation.' + uuid4().hex, shared=False, lazy=False)
    def record(**kwargs):
        calls.append((datetime.now(timezone.utc), dict(kwargs)))
        if len(calls) >= 2:
            completed.set()

    entry, due = save_entry(app, record.name, interval=2)
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=0.05)
    try:
        with worker(app):
            deadline = time.monotonic() + 10
            while not completed.is_set() and time.monotonic() < deadline:
                scheduler.tick()
                time.sleep(0.02)
            assert completed.is_set()
        assert len(calls) == 2
        assert calls[0][0] >= due
        assert (calls[1][0] - calls[0][0]).total_seconds() >= 1.8
        assert calls[0][1]['bootstrap_token'] == 'first'
        assert 'bootstrap_token' not in calls[1][1]
        stored = RedBeatSchedulerEntry.from_key(entry.key, app=app)
        assert stored.total_run_count == 2
        assert 'bootstrap_token' not in stored.kwargs
        assert stored.first_run_kwargs == {}
    finally:
        scheduler.close()
