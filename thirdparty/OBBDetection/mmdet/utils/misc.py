import os
import torch
import subprocess
import psutil

def get_visible_devices():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        return [int(dev) for dev in visible_devices.split(',')]
    else:
        return list(range(torch.cuda.device_count()))

def get_gpu_memory(visible_devices):
    """Get the current GPU usage for visible devices."""
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
            '--format=csv,nounits,noheader'
        ]).decode('utf-8')
    # Convert the result into a list of dictionaries
    gpu_memory = []
    for idx, line in enumerate(result.strip().split('\n')):
        if idx in visible_devices:
            total, used, free = line.split(',')
            gpu_memory.append({
                'total': int(total),
                'used': int(used),
                'free': int(free)
            })
    return gpu_memory

def find_best_gpu(threshold=5120, max_retries=5):
    """Find the GPU with the most free memory that is above a certain threshold."""
    visible_devices = get_visible_devices()
    torch.cuda.empty_cache()
    for _ in range(max_retries):
        gpu_memory = get_gpu_memory(visible_devices)
        best_gpu = None
        max_free_memory = 0

        for i, mem in enumerate(gpu_memory):
            if mem['free'] > max_free_memory and mem['free'] >= threshold:
                best_gpu = i
                max_free_memory = mem['free']
        
        if best_gpu is not None:
            return best_gpu

    return None

def calculate_nproc_gpu_source(single_task_mem=5120):
    """Calculate the appropriate number of parallel processes based on available GPU memory."""
    visible_devices = get_visible_devices()
    torch.cuda.empty_cache()
    gpu_memory = get_gpu_memory(visible_devices)
    total_free_memory = sum(mem['free'] for mem in gpu_memory)
    return total_free_memory // single_task_mem

def find_multi_gpu_devices(single_task_mem=5120):
    """Find the GPUs that have enough memory to run a single task."""
    visible_devices = get_visible_devices()
    torch.cuda.empty_cache()
    gpu_memory = get_gpu_memory(visible_devices)
    multi_gpu_devices = []
    for i, mem in enumerate(gpu_memory):
        for _ in range(mem['free'] // single_task_mem):
            multi_gpu_devices.append(i)
    return multi_gpu_devices

def check_gpu_memory_allocation(pid):
    """
        Check if the process with the given pid is using a GPU.
    """
    try:
        # Verify the process exists
        if not psutil.pid_exists(pid):
            return False
        
        # Run the nvidia-smi command to get the process list
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', 
                                 '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
        
        # Parse the output
        for line in result.stdout.split('\n'):
            if line:
                process_info = line.split(',')
                process_pid = int(process_info[0].strip())
                used_memory = int(process_info[1].strip())
                
                if process_pid == pid and used_memory > 512:
                    return True
        
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def is_single_gpu_available(device_id, pid, threshold=5120, max_retries=5):
    """
        Check if the GPU with device_id has enough free memory larger 
        than threshold, which is available to run a single task.
    """
    device = torch.device(f'cuda:{device_id}')
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for _ in range(max_retries):
        torch.cuda.empty_cache()
        free_memory = (total_memory - torch.cuda.memory_reserved(device)) / (1024 ** 2)
        if free_memory >= threshold:
            return True
    return False

def get_gpu_index(pid):
    """
        Get the GPU physical index for the given PID.
    """
    try:
        # Verify the process exists
        if not psutil.pid_exists(pid):
            return None
        
        # Run the nvidia-smi command and capture the output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if the command ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
        
        # Process the output to find the GPU index for the given PID
        for line in result.stdout.splitlines():
            columns = line.split()
            if len(columns) >= 5 and str(pid) == columns[4]:
                gpu_memory_usage = int(columns[-2][:-3]) # Remove the 'MiB' suffix
                gpu_index = int(columns[1])
                if gpu_memory_usage > 512:
                    return gpu_index
        
        # If the PID was not found in the nvidia-smi output
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def find_gpu_memory_allocation(pid):
    """
        Find the GPU logical index within the visible devices set, and the process with the given 
        pid has been allocated with the GPU memory on that device.
    """
    try:
        physical_id = get_gpu_index(pid)
        if physical_id is None:
            return None
        visible_devices = get_visible_devices()
        device_id = visible_devices.index(physical_id)
        return device_id
    except Exception as e:
        print(f"Error: {e}")
        return None
