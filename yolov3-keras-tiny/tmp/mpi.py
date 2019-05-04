# @Time    : 2019/4/16 14:48
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : mpi.py
import os
import re

rpm_dir_source = r'D:\软件\CentOS-7-x86_64-DVD-1611\Packages'


def filter_rpm_file():
    packages = ['libibverbs-utils', 'libibverbs-devel', 'libibverbs-devel-static', 'libmlx4', 'libmlx5', 'ibutils',
                'libibcm', 'libibcommon', 'libibmad', 'libibumad', 'rdma', ' librdmacm-utils', 'librdmacm-devel',
                'librdmacm', 'libibumad-devel', 'perftest']
    items = os.listdir(rpm_dir_source)
    rpm_filtered = list()
    result = list()
    for package in packages:
        for item in items:
            if re.search('^%s-\d' % (package), item):
                rpm_filtered.append(item)
                result.append(package)
                print(package, item)
                break
    print(result)
    print(set(packages)-set(result))
    return rpm_filtered


def bin_files():
    bin_dir = r'D:\inspur任务\aistation1.1\后台代码\c- code\script\dockermanage\infinibandlib\bin'
    items = os.listdir(bin_dir)
    print(items)
    print(len(items))


if __name__ == '__main__':
    bin_files()
