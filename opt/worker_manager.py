import os
import shutil
import time
from multiprocessing import Process

ERROR_CODE = 'eval_error'


class WorkerManager(object):

    def __init__(self, worker_ids, poll_time):

        self.worker_ids = worker_ids
        self.num_workers = len(self.worker_ids)
        self.poll_time = poll_time

        self.latest_results = []

        self.reset()

    def reset(self):

        self._child_reset()

    def _child_reset(self):
        raise NotImplementedError('Implement a child of this class')

    def fetch_latest_results(self):
        ret = self.latest_results
        self.latest_results = []

        return ret

    def close_all(self):
        raise NotImplementedError('Implement a child of this class')

    def one_worker_is_free(self):
        raise NotImplementedError('Implement a child of this class')

    def all_workers_are_free(self):
        raise NotImplementedError('Implement a child of this class')

    def _dispatch_job(self, func_caller, point, qinfo, worker_id, **kwargs):
        raise NotImplementedError('Implement a child of this class')

    def dispatch_single_job(self, func_caller, point, qinfo, **kwargs):
        raise NotImplementedError('Implement a child of this class')

    def dispatch_batch_of_jobs(self, func_caller, points, qinfos, **kwargs):
        raise NotImplementedError('Implement a child of this class')


class GPUWorkerManager(WorkerManager):

    def __init__(self, gpu_ids, tmp_dir, log_dir, poll_time=0.5):
        super(GPUWorkerManager, self).__init__(gpu_ids, poll_time)
        self.tmp_dir = tmp_dir
        self.log_dir = log_dir
        self.start_time = time.time()
        self._directory_set_up()
        self._child_reset()

    def _directory_set_up(self):
        """ Set up the working directories for storing results in """
        # Results dirs.Not using working dirs as I want to preserve the results for tensorboard
        self.results_dir_names = {wid:f'{self.tmp_dir}/results_{wid}' for wid in self.worker_ids}
        # Last time received from a worker
        self.last_receive_times = {wid:0.0 for wid in self.worker_ids}
        # What to call a results file
        self._results_file = 'results.txt'
        self._num_file_read_attmpts = 10

    @classmethod
    def _delete_dirs(cls, list_of_dir_names):
        """ Deletes a list of directories. """
        for dir_name in list_of_dir_names:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)

    @classmethod
    def _delete_and_create_dirs(cls, list_of_dir_names):
        """ Deletes a list of directories and creates new ones. """
        for dir_name in list_of_dir_names:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)

    def _child_reset(self):
        # If haven't had set up done yet
        if not hasattr(self, 'results_dir_names'):  # Just for the super constructor.
            return

        self._delete_and_create_dirs(self.results_dir_names.values())
        self.free_workers = set(self.worker_ids)
        self.qinfos_in_progress = {wid: None for wid in self.worker_ids}
        self.worker_processes = {wid: None for wid in self.worker_ids}

    def _get_result_file_name_for_worker(self, worker_id):
        """ Computes the result file name for the worker. """
        return os.path.join(self.results_dir_names[worker_id], self._results_file)

    def _read_results_from_file(self, results_file_name):
        """ Read the results from a file """
        num_attempts = 0
        while num_attempts < self._num_file_read_attmpts:
            try:
                file_reader = open(results_file_name, 'r')
                read_in = file_reader.read().strip()
                try:
                    # Try to read as a float, likely an error if not from file in use or something
                    read_in = float(read_in)
                    result = read_in
                except:
                    pass
                file_reader.close()
                break
            except:
                time.sleep(self.poll_time)
                file_reader.close()
                result = ERROR_CODE

        return result

    def _read_worker_results_and_update(self, worker_id):
        """ Reads worker ile status and updates the tracking """
        results_file_name = self._get_result_file_name_for_worker(worker_id)
        val = self._read_results_from_file(results_file_name)
        # Update the relevant data store
        qinfo = self.qinfos_in_progress[worker_id]
        qinfo.val = val
        # if not hasattr(qinfo, 'true_val'):
        #     qinfo.true_val = val
        qinfo.receive_time = time.time() - self.start_time # self.optimiser.get_curr_spent_capital()
        qinfo.eval_time = qinfo.receive_time - qinfo.send_time
        self.latest_results.append(qinfo)
        # Update receive time
        self.last_receive_times[worker_id] = qinfo.receive_time
        # Delete the file.
        os.remove(results_file_name)
        # Delete content in a working directory.
        # shutil.rmtree(self.working_dir_names[worker_id])
        # Add the worker to the list of free workers and clear qinfos in progress.
        self.worker_processes[worker_id].terminate()
        self.worker_processes[worker_id] = None
        self.qinfos_in_progress[worker_id] = None
        self.free_workers.add(worker_id)

    def _worker_is_free(self, worker_id):
        """ Checks if worker with worker_id is free. """
        if worker_id in self.free_workers:
            return True
        worker_result_file_name = self._get_result_file_name_for_worker(worker_id)
        if os.path.exists(worker_result_file_name): # Results file exists if worker has finished
            self._read_worker_results_and_update(worker_id)
        else:
            return False

    def close_all(self):
        pass

    def _get_last_receive_time(self):
        """ Returns the last time we received a job. """
        all_receive_times = self.last_receive_times.values()
        return max(all_receive_times)

    def one_worker_is_free(self):
        """ Returns true if a worker is free. """
        for wid in self.worker_ids:
            if self._worker_is_free(wid):
                return self._get_last_receive_time()
        return None

    def all_workers_are_free(self):
        """ Returns true if all workers are free. """
        all_are_free = True
        for wid in self.worker_ids:
            all_are_free = self._worker_is_free(wid) and all_are_free
        if all_are_free:
            return self._get_last_receive_time()
        else:
            return None

    def _dispatch_job(self, func_caller, point, qinfo, worker_id, **kwargs):
        """ Dispatches evaluation to worker_id. """
        # pylint: disable=star-args
        if self.qinfos_in_progress[worker_id] is not None:
            err_msg = 'qinfos_in_progress: %s,\nfree_workers: %s.' % (
                str(self.qinfos_in_progress), str(self.free_workers))
            print(err_msg)
            raise ValueError('Check if worker is free before sending evaluation.')
        # First add all the data to qinfo
        qinfo.worker_id = worker_id
        qinfo.log_dir = self.log_dir
        qinfo.result_file = self._get_result_file_name_for_worker(worker_id)
        qinfo.point = point
        qinfo.send_time = time.time() - self.start_time # Should be abstracted to the searcher
        # Create the working directory
        # os.makedir(qinfo.log_dir) # will create this later as we want model specific, constant directories
        # Dispatch the evaluation in a new process
        target_func = lambda: func_caller.eval_single(point, qinfo, **kwargs)
        # target_func()
        self.worker_processes[worker_id] = Process(target=target_func)
        self.worker_processes[worker_id].start()
        time.sleep(3)
        # Add the qinfo to the in progress bar and remove from free_workers
        self.qinfos_in_progress[worker_id] = qinfo
        self.free_workers.discard(worker_id)

    def dispatch_single_job(self, func_caller, point, qinfo, **kwargs):
        worker_id = self.free_workers.pop()
        self._dispatch_job(func_caller, point, qinfo, worker_id, **kwargs)

    def dispatch_batch_of_jobs(self, func_caller, points, qinfos, **kwargs):
        assert len(points) == self.num_workers
        for idx in range(self.num_workers):
            self._dispatch_job(func_caller, points[idx], qinfos[idx], self.worker_ids[idx], **kwargs)

    def close_all_jobs(self):
        """ Closes all jobs. """
        pass


