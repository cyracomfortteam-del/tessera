//! Admission control for the gateway.
//!
//! A `tokio::sync::Semaphore` bounds how many requests are in flight at once (the serving
//! "batch budget"); everything else waits. Atomic counters expose live queue/run depth for
//! the `/health` endpoint. This is the Rust-side mirror of the Python continuous-batching
//! scheduler's admission logic — back-pressure instead of unbounded concurrency.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Semaphore;

pub struct Scheduler {
    permits: Arc<Semaphore>,
    max_concurrency: usize,
    queued: AtomicUsize,
    running: AtomicUsize,
}

impl Scheduler {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(max_concurrency)),
            max_concurrency,
            queued: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
        }
    }

    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    pub fn running(&self) -> usize {
        self.running.load(Ordering::Relaxed)
    }

    pub fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    /// Run `fut` under the concurrency budget: count it as queued until a permit is free,
    /// then as running for its duration.
    pub async fn admit<F, T>(&self, fut: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        self.queued.fetch_add(1, Ordering::Relaxed);
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore");
        self.queued.fetch_sub(1, Ordering::Relaxed);

        self.running.fetch_add(1, Ordering::Relaxed);
        let result = fut.await;
        self.running.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        result
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn admit_never_exceeds_budget() {
        let sched = Arc::new(Scheduler::new(2));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..12 {
            let sched = sched.clone();
            let peak = peak.clone();
            handles.push(tokio::spawn(async move {
                let s = sched.clone();
                sched
                    .admit(async move {
                        peak.fetch_max(s.running(), Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_millis(15)).await;
                    })
                    .await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert!(
            peak.load(Ordering::Relaxed) <= 2,
            "concurrency exceeded the budget"
        );
        assert_eq!(sched.running(), 0);
        assert_eq!(sched.queued(), 0);
    }
}
