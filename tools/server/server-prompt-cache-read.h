#pragma once

#include <string>
#include <vector>
#include <algorithm>

// one in-flight disk read. a second request for the same path joins it. a second path
// waits. the server thread does not fread; it only learns pending / ready / failed.

enum class server_prompt_cache_read_reentry { absent, pending, ready, failed };

enum class server_prompt_cache_read_kind { pending, ready, failed };

struct server_prompt_cache_read_arm {
    server_prompt_cache_read_kind kind  = server_prompt_cache_read_kind::pending;
    bool                          start = false;
};

struct server_prompt_cache_read_take {
    bool        ok   = false;
    bool        last = false;
    std::string path;
};

struct server_prompt_cache_read_job {
    std::string       path;
    bool              inflight = false;
    bool              done     = false;
    bool              failed   = false;
    std::vector<int>  waiters;
    int               refs     = 0;
};

struct server_prompt_cache_read_sched {
    std::vector<server_prompt_cache_read_job> jobs;

    server_prompt_cache_read_reentry reentry(int task_id) const {
        const server_prompt_cache_read_job * job = find_waiter(task_id);

        if (job == nullptr) {
            return server_prompt_cache_read_reentry::absent;
        }
        if (!job->done) {
            return server_prompt_cache_read_reentry::pending;
        }

        return job->failed ? server_prompt_cache_read_reentry::failed : server_prompt_cache_read_reentry::ready;
    }

    server_prompt_cache_read_arm arm(const std::string & path, int task_id) {
        server_prompt_cache_read_arm out;

        for (auto & job : jobs) {
            if (job.path != path) {
                continue;
            }

            job.waiters.push_back(task_id);

            if (job.done && job.failed) {
                job.refs++;
                out.kind = server_prompt_cache_read_kind::failed;

                return out;
            }
            if (job.done) {
                job.refs++;
                out.kind = server_prompt_cache_read_kind::ready;

                return out;
            }

            out.kind = server_prompt_cache_read_kind::pending;

            return out;
        }

        server_prompt_cache_read_job job;
        job.path = path;
        job.waiters.push_back(task_id);
        job.inflight = !busy();
        jobs.push_back(std::move(job));

        out.kind  = server_prompt_cache_read_kind::pending;
        out.start = jobs.back().inflight;

        return out;
    }

    std::vector<int> complete(const std::string & path, bool ok) {
        std::vector<int> wake;

        for (auto it = jobs.begin(); it != jobs.end(); ++it) {
            if (it->path != path || it->done) {
                continue;
            }

            it->inflight = false;
            it->done     = true;
            it->failed   = !ok;
            it->refs     = (int) it->waiters.size();
            wake         = it->waiters;

            if (it->waiters.empty()) {
                jobs.erase(it);
            }

            break;
        }

        return wake;
    }

    std::string next_path() {
        if (busy()) {
            return "";
        }

        for (auto & job : jobs) {
            if (!job.done && !job.inflight) {
                job.inflight = true;

                return job.path;
            }
        }

        return "";
    }

    bool cancel(int task_id) {
        for (auto it = jobs.begin(); it != jobs.end(); ++it) {
            auto w = std::find(it->waiters.begin(), it->waiters.end(), task_id);
            if (w == it->waiters.end()) {
                continue;
            }

            it->waiters.erase(w);
            if (it->done && it->refs > 0) {
                it->refs--;
            }
            if (it->waiters.empty() && (!it->inflight || it->done) && it->refs <= 0) {
                jobs.erase(it);
            }

            return true;
        }

        return false;
    }

    server_prompt_cache_read_take take(int task_id) {
        server_prompt_cache_read_take out;

        for (auto it = jobs.begin(); it != jobs.end(); ++it) {
            if (!it->done || it->failed) {
                continue;
            }

            auto w = std::find(it->waiters.begin(), it->waiters.end(), task_id);
            if (w == it->waiters.end()) {
                continue;
            }

            it->waiters.erase(w);
            if (it->refs > 0) {
                it->refs--;
            }

            out.ok   = true;
            out.last = it->refs <= 0 && it->waiters.empty();
            out.path = it->path;

            if (out.last) {
                jobs.erase(it);
            }

            return out;
        }

        return out;
    }

private:
    bool busy() const {
        for (const auto & job : jobs) {
            if (job.inflight) {
                return true;
            }
        }

        return false;
    }

    const server_prompt_cache_read_job * find_waiter(int task_id) const {
        for (const auto & job : jobs) {
            if (std::find(job.waiters.begin(), job.waiters.end(), task_id) != job.waiters.end()) {
                return &job;
            }
        }

        return nullptr;
    }
};
