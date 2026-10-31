//! Per-model admission and idle residency. Leases survive cancelled HTTP futures.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

pub struct Residency<E> {
    engine: Arc<AsyncMutex<Option<E>>>,
    activity: Arc<Activity>,
    admission: Arc<Semaphore>,
    idle_timeout: Duration,
}

struct Activity {
    requests: AtomicUsize,
    finished: Mutex<Instant>,
}

struct RequestLease {
    activity: Arc<Activity>,
    _permit: OwnedSemaphorePermit,
}

pub struct EngineLease<E> {
    engine: OwnedMutexGuard<Option<E>>,
    _request: RequestLease,
}

impl<E> Residency<E> {
    pub fn new(idle_timeout: Duration, maximum_requests: usize) -> Self {
        Self {
            engine: Arc::new(AsyncMutex::new(None)),
            activity: Arc::new(Activity {
                requests: AtomicUsize::new(0),
                finished: Mutex::new(Instant::now()),
            }),
            admission: Arc::new(Semaphore::new(maximum_requests)),
            idle_timeout,
        }
    }

    pub async fn acquire(&self) -> Result<EngineLease<E>> {
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| anyhow!("model request queue is full"))?;
        self.activity.requests.fetch_add(1, Ordering::SeqCst);
        let request = RequestLease {
            activity: self.activity.clone(),
            _permit: permit,
        };
        let engine = self.engine.clone().lock_owned().await;
        Ok(EngineLease {
            engine,
            _request: request,
        })
    }

    pub fn evict_idle(&self, now: Instant) -> bool {
        if self.idle_timeout.is_zero() {
            return false;
        }
        let Ok(mut engine) = self.engine.try_lock() else {
            return false;
        };
        if self.activity.requests.load(Ordering::SeqCst) != 0 {
            return false;
        }
        let Ok(finished) = self.activity.finished.lock() else {
            return false;
        };
        if now.saturating_duration_since(*finished) < self.idle_timeout {
            return false;
        }
        engine.take().is_some()
    }

    pub fn state(&self) -> &'static str {
        match self.engine.try_lock() {
            Ok(engine) if engine.is_some() => "ready",
            Ok(_) => "unloaded",
            Err(_) => "busy",
        }
    }
}

impl<E> EngineLease<E> {
    pub fn engine(&mut self) -> &mut Option<E> {
        &mut self.engine
    }
}

impl Drop for RequestLease {
    fn drop(&mut self) {
        // Timestamp precedes the decrement: an eviction seeing zero requests
        // must also see a completion timestamp for this request.
        if let Ok(mut finished) = self.activity.finished.lock() {
            *finished = Instant::now();
        }
        self.activity.requests.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn serializes_loads_and_keeps_active_engines_resident() {
        let residency = Arc::new(Residency::new(Duration::from_secs(10), 2));
        let mut first = residency.acquire().await.unwrap();
        *first.engine() = Some(42);
        let other = residency.clone();
        let second = tokio::spawn(async move {
            let mut lease = other.acquire().await.unwrap();
            assert_eq!(lease.engine().as_ref(), Some(&42));
        });
        assert!(!residency.evict_idle(Instant::now() + Duration::from_secs(100)));
        drop(first);
        second.await.unwrap();
        assert!(!residency.evict_idle(Instant::now()));
        assert!(residency.evict_idle(Instant::now() + Duration::from_secs(11)));
        assert_eq!(residency.state(), "unloaded");
        assert!(residency.acquire().await.unwrap().engine().is_none());
    }

    #[tokio::test]
    async fn admission_is_bounded_and_released_when_waiter_is_cancelled() {
        let residency = Arc::new(Residency::<usize>::new(Duration::ZERO, 1));
        let first = residency.acquire().await.unwrap();
        assert!(residency.acquire().await.is_err());
        drop(first);
        assert!(residency.acquire().await.is_ok());
        assert!(!residency.evict_idle(Instant::now() + Duration::from_secs(100)));
    }

    #[tokio::test]
    async fn cancelled_waiter_does_not_keep_activity_alive() {
        let residency = Arc::new(Residency::new(Duration::from_secs(1), 2));
        let mut first = residency.acquire().await.unwrap();
        *first.engine() = Some(7);
        let other = residency.clone();
        let waiter = tokio::spawn(async move { other.acquire().await });
        tokio::task::yield_now().await;
        waiter.abort();
        let _ = waiter.await;
        drop(first);
        assert!(residency.evict_idle(Instant::now() + Duration::from_secs(2)));
    }
}
