#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  // for size_t

typedef void (*thread_task_func)(void*);

typedef struct Task {
    thread_task_func func;
    void* arg;
} Task;

typedef struct ggml_backend_tp_threadpool {
    std::vector<std::thread> workers;
    std::queue<Task> task_queue;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
} ggml_backend_tp_threadpool;


/**
 * Create and initialize a thread pool.
 * 
 * @param num_threads Number of threads to create in the pool.
 * @return Pointer to the initialized ThreadPool, or NULL on failure.
 */
void ggml_backend_tp_threadpool_init(ggml_backend_tp_threadpool* pool, size_t num_threads);

/**
 * Enqueue a task to the thread pool.
 * 
 * @param pool Pointer to a ThreadPool.
 * @param func Function pointer for the task.
 * @param arg Argument to be passed to the task function.
 */
void ggml_backend_tp_threadpool_enqueue(ggml_backend_tp_threadpool* pool, thread_task_func func, void* arg);

/**
 * Destroy the thread pool and free its resources.
 * 
 * @param pool Pointer to the ThreadPool to destroy.
 */
void ggml_backend_tp_threadpool_destroy(ggml_backend_tp_threadpool* pool);


typedef struct ggml_backend_tp_semaphore {
    std::mutex mutex;
    std::condition_variable cv;
    int count;
};

void ggml_backend_tp_semaphore_release(ggml_backend_tp_semaphore * semaphore, int n = 1);
void ggml_backend_tp_semaphore_acquire(ggml_backend_tp_semaphore * semaphore);
#ifdef __cplusplus
}
#endif

#endif // THREAD_POOL_H
