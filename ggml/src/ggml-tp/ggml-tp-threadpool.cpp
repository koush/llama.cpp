#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdlib>
#include "ggml-tp-threadpool.h"

static void thread_pool_worker(ggml_backend_tp_threadpool* pool) {
    while (true) {
        Task task;

        {
            std::unique_lock<std::mutex> lock(pool->queue_mutex);
            pool->condition.wait(lock, [&] {
                return pool->stop || !pool->task_queue.empty();
            });

            if (pool->stop && pool->task_queue.empty())
                return;

            task = pool->task_queue.front();
            pool->task_queue.pop();
        }

        if (task.func)
            task.func(task.arg);
    }
}

void ggml_backend_tp_threadpool_init(ggml_backend_tp_threadpool* pool, size_t num_threads) {
    pool->stop = false;

    for (size_t i = 0; i < num_threads; ++i) {
        pool->workers.emplace_back([pool] {
            thread_pool_worker(pool);
        });
    }
}

void ggml_backend_tp_threadpool_enqueue(ggml_backend_tp_threadpool* pool, thread_task_func func, void* arg) {
    {
        std::unique_lock<std::mutex> lock(pool->queue_mutex);
        pool->task_queue.push(Task{func, arg});
    }
    pool->condition.notify_one();
}

void ggml_backend_tp_threadpool_destroy(ggml_backend_tp_threadpool* pool) {
    {
        std::unique_lock<std::mutex> lock(pool->queue_mutex);
        pool->stop = true;
    }
    pool->condition.notify_all();

    for (std::thread& worker : pool->workers) {
        if (worker.joinable())
            worker.join();
    }

    pool->workers.clear();
}


void ggml_backend_tp_semaphore_release(ggml_backend_tp_semaphore * semaphore, int n) {
    std::unique_lock<std::mutex> lock(semaphore->mutex);
    semaphore->count += n;
    for (int i = 0; i < n; ++i) {
        semaphore->cv.notify_one();
    }
}

void ggml_backend_tp_semaphore_acquire(ggml_backend_tp_semaphore * semaphore) {
    std::unique_lock<std::mutex> lock(semaphore->mutex);
    semaphore->cv.wait(lock, [semaphore]() { return semaphore->count > 0; });
    --semaphore->count;
}
