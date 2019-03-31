from tensorboardX import SummaryWriter
from subprocess import check_output
from psutil import cpu_percent, virtual_memory, swap_memory


class GlobalSummaryWriter(SummaryWriter):

	def __get_gpu_memory_usage(self):
		result = check_output(
			[
				'nvidia-smi', '--query-gpu=memory.used',
				'--format=csv,nounits,noheader'
			], encoding='utf-8')
		# Convert lines into a dictionary
		gpu_memory = [int(x) for x in result.strip().split('\n')]
		gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
		return gpu_memory_map[0]

	def __get_gpu_utilization(self):
		"""Get the current gpu usage.

		Returns
		-------
		usage: dict
			Keys are device ids as integers.
			Values are memory usage as integers in MB.
		"""
		result = check_output(
			[
				'nvidia-smi', '--query-gpu=utilization.gpu',
				'--format=csv,nounits,noheader'
			], encoding='utf-8')
		# Convert lines into a dictionary
		gpu_memory = [int(x) for x in result.strip().split('\n')]
		gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
		return gpu_memory_map[0]

	def __get_cpu_utilization(self):
		return cpu_percent()

	def __get_memory_utilization_percentage(self):
		return virtual_memory().percent

	def __get_swap_memory_utilization_percentage(self):
		return swap_memory().percent

	def log_system_metrics(self, global_step=None):
		self.add_scalar('System/GPUMemoryUtilization', self.__get_gpu_memory_usage(), global_step=global_step)
		self.add_scalar('System/GPUUtilization', self.__get_gpu_utilization(), global_step=global_step)
		self.add_scalar('System/CPUUtilization', self.__get_cpu_utilization(), global_step=global_step)
		self.add_scalar('System/MemoryUsagePercentage', self.__get_memory_utilization_percentage(),
						global_step=global_step)
		self.add_scalar('System/SwapMemoryUsagePercentage', self.__get_swap_memory_utilization_percentage(),
						global_step=global_step)
