#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器DRAM内存带宽测试工具
基于STREAM-like基准测试，使用numpy底层优化和多进程并行
"""

import time
import sys
import os
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count


class MemoryBandwidthTester:
    """内存带宽测试器 - STREAM-like基准"""

    def __init__(self, array_size_mb=100, iterations=10, dtype=np.float64):
        """
        参数:
            array_size_mb: 每个数组的大小(MB)，建议大于L3缓存的4倍
            iterations: 每个测试的迭代次数
            dtype: 数据类型，float64(8字节)或float32(4字节)
        """
        self.array_size_mb = array_size_mb
        self.iterations = iterations
        self.dtype = dtype
        self.bytes_per_element = np.dtype(dtype).itemsize
        # 计算元素个数
        self.n_elements = (array_size_mb * 1024 * 1024) // self.bytes_per_element

    def _create_arrays(self):
        """创建对齐的numpy数组"""
        a = np.ones(self.n_elements, dtype=self.dtype)
        b = np.ones(self.n_elements, dtype=self.dtype) * 2.0
        c = np.ones(self.n_elements, dtype=self.dtype) * 3.0
        scalar = np.dtype(self.dtype).type(1.5)
        return a, b, c, scalar

    def test_copy(self, _=None):
        """COPY: A = B (读取B，写入A) -> 2次内存访问"""
        a, b, c, scalar = self._create_arrays()

        # 预热
        for _ in range(3):
            a[:] = b[:]

        start = time.perf_counter()
        for _ in range(self.iterations):
            a[:] = b[:]
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = 2 * self.n_elements * self.bytes_per_element * self.iterations
        bandwidth_gb_s = (total_bytes / elapsed) / (1024**3)
        return {
            'test': 'COPY',
            'bandwidth_gb_s': bandwidth_gb_s,
            'elapsed_s': elapsed,
            'total_gb': total_bytes / (1024**3)
        }

    def test_scale(self, _=None):
        """SCALE: A = scalar * B (读取B，写入A) -> 2次内存访问"""
        a, b, c, scalar = self._create_arrays()

        for _ in range(3):
            a[:] = scalar * b[:]

        start = time.perf_counter()
        for _ in range(self.iterations):
            a[:] = scalar * b[:]
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = 2 * self.n_elements * self.bytes_per_element * self.iterations
        bandwidth_gb_s = (total_bytes / elapsed) / (1024**3)
        return {
            'test': 'SCALE',
            'bandwidth_gb_s': bandwidth_gb_s,
            'elapsed_s': elapsed,
            'total_gb': total_bytes / (1024**3)
        }

    def test_add(self, _=None):
        """ADD: A = B + C (读取B、C，写入A) -> 3次内存访问"""
        a, b, c, scalar = self._create_arrays()

        for _ in range(3):
            a[:] = b[:] + c[:]

        start = time.perf_counter()
        for _ in range(self.iterations):
            a[:] = b[:] + c[:]
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = 3 * self.n_elements * self.bytes_per_element * self.iterations
        bandwidth_gb_s = (total_bytes / elapsed) / (1024**3)
        return {
            'test': 'ADD',
            'bandwidth_gb_s': bandwidth_gb_s,
            'elapsed_s': elapsed,
            'total_gb': total_bytes / (1024**3)
        }

    def test_triad(self, _=None):
        """TRIAD: A = B + scalar * C (读取B、C，写入A) -> 3次内存访问"""
        a, b, c, scalar = self._create_arrays()

        for _ in range(3):
            a[:] = b[:] + scalar * c[:]

        start = time.perf_counter()
        for _ in range(self.iterations):
            a[:] = b[:] + scalar * c[:]
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = 3 * self.n_elements * self.bytes_per_element * self.iterations
        bandwidth_gb_s = (total_bytes / elapsed) / (1024**3)
        return {
            'test': 'TRIAD',
            'bandwidth_gb_s': bandwidth_gb_s,
            'elapsed_s': elapsed,
            'total_gb': total_bytes / (1024**3)
        }

    def run_single_thread(self):
        """单线程测试"""
        print(f"\n{'='*60}")
        print(f"单线程内存带宽测试")
        print(f"{'='*60}")
        print(f"数组大小: {self.array_size_mb} MB x 3 arrays")
        print(f"数据类型: {self.dtype.__name__} ({self.bytes_per_element} bytes/element)")
        print(f"迭代次数: {self.iterations}")
        print(f"CPU: {cpu_count()} 核心")
        print(f"{'-'*60}")

        results = []
        for test_func in [self.test_copy, self.test_scale, self.test_add, self.test_triad]:
            result = test_func()
            results.append(result)
            print(f"{result['test']:8s}: {result['bandwidth_gb_s']:8.2f} GB/s  "
                  f"({result['total_gb']:.2f} GB in {result['elapsed_s']:.3f}s)")

        avg_bw = sum(r['bandwidth_gb_s'] for r in results) / len(results)
        print(f"{'-'*60}")
        print(f"平均带宽: {avg_bw:.2f} GB/s")
        return results

    def run_multi_process(self, num_processes=None):
        """多进程并行测试（绕过GIL，更接近理论峰值）"""
        if num_processes is None:
            num_processes = cpu_count()

        print(f"\n{'='*60}")
        print(f"多进程并行内存带宽测试")
        print(f"{'='*60}")
        print(f"进程数: {num_processes} / {cpu_count()} 核心")
        print(f"每个进程数组: {self.array_size_mb} MB x 3")
        print(f"总内存占用: ~{self.array_size_mb * 3 * num_processes} MB")
        print(f"{'-'*60}")

        results = []
        test_funcs = [self.test_copy, self.test_scale, self.test_add, self.test_triad]

        for test_func in test_funcs:
            with Pool(processes=num_processes) as pool:
                start = time.perf_counter()
                proc_results = pool.map(test_func, range(num_processes))
                end = time.perf_counter()

            total_bw = sum(r['bandwidth_gb_s'] for r in proc_results)
            test_name = proc_results[0]['test']
            results.append({
                'test': test_name,
                'bandwidth_gb_s': total_bw,
                'per_core_gb_s': total_bw / num_processes,
                'elapsed_s': end - start
            })
            print(f"{test_name:8s}: {total_bw:8.2f} GB/s  "
                  f"(每核心: {total_bw/num_processes:.2f} GB/s)")

        avg_bw = sum(r['bandwidth_gb_s'] for r in results) / len(results)
        print(f"{'-'*60}")
        print(f"总平均带宽: {avg_bw:.2f} GB/s")
        return results

    def run_write_bandwidth(self):
        """纯写入带宽测试（memset-like）"""
        print(f"\n{'='*60}")
        print(f"纯写入带宽测试 (memset-like)")
        print(f"{'='*60}")

        a = np.ones(self.n_elements, dtype=self.dtype)

        for _ in range(3):
            a.fill(1.0)

        start = time.perf_counter()
        for _ in range(self.iterations):
            a.fill(1.0)
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = self.n_elements * self.bytes_per_element * self.iterations
        bw = (total_bytes / elapsed) / (1024**3)
        print(f"WRITE: {bw:.2f} GB/s")
        return bw

    def run_read_bandwidth(self):
        """纯读取带宽测试（sum-like，避免被优化掉）"""
        print(f"\n{'='*60}")
        print(f"纯读取带宽测试 (sum-like)")
        print(f"{'='*60}")

        a = np.ones(self.n_elements, dtype=self.dtype)

        # 使用结果避免编译器优化掉读取操作
        results = []
        for _ in range(3):
            results.append(np.sum(a))

        start = time.perf_counter()
        for _ in range(self.iterations):
            results.append(np.sum(a))
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = self.n_elements * self.bytes_per_element * self.iterations
        bw = (total_bytes / elapsed) / (1024**3)
        print(f"READ:  {bw:.2f} GB/s")
        # 防止优化
        _ = sum(results)
        return bw


def estimate_optimal_size():
    """根据系统内存估算合适的测试数组大小"""
    try:
        import psutil
        total_mem = psutil.virtual_memory().total
        # 使用总内存的5%或至少100MB
        size_mb = max(100, int(total_mem * 0.05 / (1024*1024)))
        # 但不超过1GB避免测试时间过长
        return min(size_mb, 1024)
    except ImportError:
        return 256


def print_system_info():
    """打印系统信息"""
    print(f"\n{'#'*60}")
    print(f"# 系统信息")
    print(f"{'#'*60}")
    print(f"CPU核心数: {cpu_count()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"总内存: {mem.total / (1024**3):.1f} GB")
        print(f"可用内存: {mem.available / (1024**3):.1f} GB")
    except ImportError:
        print("安装 psutil 可获取更详细的内存信息: pip install psutil")

    # 尝试读取CPU信息
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    print(f"CPU型号: {line.split(':')[1].strip()}")
                    break
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='服务器DRAM内存带宽测试')
    parser.add_argument('-s', '--size', type=int, default=None,
                        help='每个数组大小(MB)，默认自动估算')
    parser.add_argument('-i', '--iterations', type=int, default=20,
                        help='迭代次数，默认20')
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='多进程数，默认使用所有核心')
    parser.add_argument('--single-only', action='store_true',
                        help='仅运行单线程测试')
    parser.add_argument('--multi-only', action='store_true',
                        help='仅运行多进程测试')
    parser.add_argument('--float32', action='store_true',
                        help='使用float32而非float64')

    args = parser.parse_args()

    array_size = args.size or estimate_optimal_size()
    dtype = np.float32 if args.float32 else np.float64

    print_system_info()

    tester = MemoryBandwidthTester(
        array_size_mb=array_size,
        iterations=args.iterations,
        dtype=dtype
    )

    if not args.multi_only:
        tester.run_single_thread()
        tester.run_read_bandwidth()
        tester.run_write_bandwidth()

    if not args.single_only:
        tester.run_multi_process(num_processes=args.processes)

    print(f"\n{'#'*60}")
    print(f"# 测试完成")
    print(f"# 注意: Python/numpy测试受解释器开销影响，")
    print(f"# 如需精确峰值请使用C版本的STREAM或Intel MKL")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()
