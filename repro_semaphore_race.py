"""
Reproduction for: RuntimeError: Semaphore released too many times

Root cause: A race between Timeout exceptions and semaphore
notifications in gevent's cooperative scheduler.

Scenario (matching the production bug):

- A background thread (like TagAlongThread) continuously spawns
  concurrent greenlets (like MultiObject workers) that contend on
  a shared BoundedSemaphore (the ThreadPoolExecutor's _shutdown_lock).
  Each worker acquires the lock, does minimal work, and releases it.

- The main greenlet runs a tight loop with short Timeouts (like
  _check_if_vms_is_ready calling vms.wait(timeout=10)), where it
  tries to acquire the same lock under a Timeout context.

- The race: greenlet G1 is blocked on acquire() in __enter__. The
  hub has two pending callbacks: (a) the Timeout expiration, and
  (b) a notification from another greenlet releasing the semaphore.
  When both fire in quick succession, the Timeout causes acquire()
  to raise, but the stale notification re-enters G1 via
  greenlet.switch(), making it proceed as if acquire() succeeded.
  G1's context manager then calls __exit__ -> release(), over-releasing.
"""
# from gevent import monkey
# monkey.patch_all()
import easypy
import sys
import time
import threading
import traceback
import gevent
from gevent import sleep, Timeout, spawn
from gevent.lock import BoundedSemaphore


NUM_CONTENDERS = 100
TIMEOUT_SECONDS = 0.0001
MAX_ITERATIONS = 1_000_000
REPORT_EVERY = 10_000

shared_lock = BoundedSemaphore(1)
stop_flag = False
reproduced = False


def contender():
    """
    Simulates a worker greenlet (like those in MultiObject / ThreadPoolExecutor)
    that acquires the shared lock, does minimal work (yield), then releases.
    This creates contention: many greenlets waiting on acquire() means the
    semaphore's _links list is populated with switch callbacks.
    """
    while not stop_flag:
        with shared_lock:
            sleep(0)
        sleep(0)


def timeout_loop():
    """
    Simulates the main-thread pattern:
      try:
          with Timeout(10):
              with _shutdown_lock:
                  ...
      except Timeout:
          pass

    The short timeout creates the window for the race: the greenlet
    blocks on acquire(), Timeout fires, but a stale release notification
    also switches into this greenlet.
    """
    global stop_flag, reproduced

    print(f"Running up to {MAX_ITERATIONS} timeout iterations...")
    print(f"  {NUM_CONTENDERS} contender greenlets competing for the same BoundedSemaphore")
    print(f"  timeout = {TIMEOUT_SECONDS}s")
    print()

    for i in range(MAX_ITERATIONS):
        if stop_flag:
            break
        try:
            with Timeout(TIMEOUT_SECONDS):
                with shared_lock:
                    sleep(0)
        except Timeout:
            pass
        except RuntimeError as e:
            if "Semaphore released too many times" in str(e):
                print(f"\n*** BUG REPRODUCED after {i} iterations ***")
                print(f"RuntimeError: {e}")
                traceback.print_exc()
                reproduced = True
                stop_flag = True
                return
        except Exception as e:
            print(f"\n[iteration {i}] unexpected: {e!r}", file=sys.stderr)
            traceback.print_exc()

        if i > 0 and i % REPORT_EVERY == 0:
            print(f"  iteration {i}...")

    stop_flag = True


def main():
    global stop_flag

    greenlets = []
    for _ in range(NUM_CONTENDERS):
        greenlets.append(spawn(contender))

    sleep(0.01)

    try:
        timeout_loop()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_flag = True
        gevent.joinall(greenlets, timeout=2)

    if reproduced:
        sys.exit(1)
    else:
        print(f"\nCompleted {MAX_ITERATIONS} iterations without reproducing the bug.")
        sys.exit(0)


if __name__ == "__main__":
    main()
