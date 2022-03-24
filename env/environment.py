import numpy as np
import math
import random
import copy

class Task:
    def __init__(self, task_id, cpu_quota, mem_quota):
        self.id = task_id
        self.len = 0
        self.enter_time = 0
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.cpu_quota = cpu_quota
        self.cpu_utilization = .0
        self.mem_quota = mem_quota
        self.mem_utilization = .0


class Machine:
    def __init__(self, machine_id, cpu_quota, mem_quota):
        self.id = machine_id  #
        self.cpu_quota = cpu_quota
        self.cpu_utilization = .0
        self.mem_quota = mem_quota
        self.mem_utilization = .0


class Env:
    initial_machines = []  # Used for cluster initialization

    def __init__(self, cluster_id, pa):
        self.pa = pa
        self.id = cluster_id
        self.cpu_quota = .0
        self.rur = .0
        self.mem_quota = .0
        self.rbd = .0
        self.machines = []

    def initialize(self):
        for i in range(self.pa.num_machine):
            mc = Machine(i, random.randint(10, 20), random.randint(50, 100))
            Env.initial_machines.append(mc)
        self.machines = copy.deepcopy(Env.initial_machines)

    def allocate_task(self, task, machine_id):
        if task == None and machine_id == None:
            print("Scheduling failed. The ID of the task or machine to "
                  "be scheduled is (task: %s, machine_id: %s)" % (task, machine_id))

        allocated = False
        machine_sche = self.find_machine(machine_id)

        if machine_sche == None:
            print("Scheduling failed. The scheduling node does not exist")

        if task.cpu_quota <= machine_sche.cpu_quota * (
                1 - machine_sche.cpu_utilization) and task.mem_quota <= machine_sche.mem_quota * (
                1 - machine_sche.mem_utilization):
            machine_sche.cpu_utilization = (machine_sche.cpu_quota * machine_sche.cpu_utilization
                                            + task.cpu_quota) / machine_sche.cpu_quota

            machine_sche.mem_utilization = (
                                                       machine_sche.mem_quota * machine_sche.mem_utilization + task.mem_quota) / machine_sche.mem_quota
            # print("Scheduling succeeded. Task %s was successfully assigned to machine %s" % (task.id, self.id))
            allocated = True
        # else:
        #     print(
        #         "Scheduling failed. The CPU or memory was insufficient. Task %s Demand: %s, machine %s surplus %s" % (
        #             task.id, (task.cpu_quota, task.mem_quota), machine_sche.id,
        #             (machine_sche.cpu_quota * (1 - machine_sche.cpu_utilization),
        #              machine_sche.mem_quota * (1 - machine_sche.mem_utilization))))
        return allocated

    def reset(self):
        if len(Env.initial_machines) == 0:
            self.initialize()
        else:
            self.machines = copy.deepcopy(Env.initial_machines)
        return self.observe(Task(-1, 0, 0))

    def observe(self, task):
        ob = []
        for mc in self.machines:
            ob.append(mc.cpu_utilization)
            ob.append(mc.mem_utilization)
        ob.append(task.cpu_quota / 20.0)
        ob.append(task.mem_quota / 20.0)
        return ob

    def step(self, task, machine_id, next_task):
        al_res = self.allocate_task(task, machine_id)
        next_obs = self.observe(next_task)
        reward = 0
        if al_res == True:
            reward = self.reward()
        return next_obs, reward, not al_res

    def reward(self):
        global alpha
        mcs_res_utilization, mcs_res_balance = [], []
        for mc in self.machines:
            res_uti_avg = (mc.cpu_utilization + mc.mem_utilization) / 2
            res_blc = 1 - math.sqrt(
                math.pow(mc.cpu_utilization - res_uti_avg, 2) + math.pow(mc.mem_utilization - res_uti_avg, 2))
            mcs_res_utilization.append(res_uti_avg)
            mcs_res_balance.append(res_blc)
        rur, rbd = np.average(mcs_res_utilization), np.average(mcs_res_balance)
        return self.pa.alpha * rur + (1 - self.pa.alpha) * rbd

    def find_machine(self, machine_id):
        rst = None
        for mc in self.machines:
            if mc.id == machine_id:
                rst = mc
                break
        return rst

    def generate_task_sequences(self, num_task):
        tasks = []
        for task_id in range(num_task):
            tasks.append(Task(task_id, 2*round(random.random(), 2), 8 * round(random.random(), 2)))
        return tasks
