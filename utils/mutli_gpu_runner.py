import time

from argparse import Namespace

from opt.worker_manager import GPUWorkerManager


class MultiGPURunner():

    def __init__(self, func_caller, gpu_ids, tmp_dir, log_dir):

        self.func_caller = func_caller
        self.worker_manager = GPUWorkerManager(gpu_ids, tmp_dir, log_dir)
        self.points_left = []
        print(f'Starting Multi GPU experiment with GPUs {gpu_ids}')

    def wait_till_free(self):
        keep_looping = True
        while keep_looping:
            last_receive_time = self.worker_manager.one_worker_is_free()
            if last_receive_time is not None:
                # Get the latest set of results and dispatch the next job.
                # self.set_curr_spent_capital(last_receive_time)
                latest_results = self.worker_manager.fetch_latest_results()
                for qinfo_result in latest_results:
                    print(f'Finished model on gpu {qinfo_result.worker_id}: {qinfo_result}')
                keep_looping = False
            else:
                time.sleep(0.5)

    def run_points(self, points):
        self.points_left = points

        while len(self.points_left) > 0:
            self.wait_till_free()
            point = self.points_left.pop()
            qinfo = Namespace(point=point, send_time=time.time())
            self.worker_manager.dispatch_single_job(self.func_caller, point, qinfo)
