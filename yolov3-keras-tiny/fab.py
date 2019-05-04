# @Time    : 2019/4/15 10:01
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : fab.py
import os
from fabric import Connection
from fabric import SerialGroup as Group


def add_new_node_ssh():
    nodes_new = {'172.17.0.12': '111111'}
    node_num = 3
    with open('/etc/hosts', 'a') as f:
        for ip, pwd in node_new.items():
            f.write('%s    node%s' % (ip, node_num))
            node_num += 1
            c = Connection(ip, port=22, user='root', connect_kwargs={'password': pwd})
            c.get('/root/.ssh/id_rsa.pub', '/root/.ssh/id_rsa.pub.bak')
            c.local('cat /root/.ssh/id_rsa.pub.bak >> /root/.ssh/authorized_keys')
            c.local('rm -f /root/.ssh/id_rsa.pub.bak')
    nodes.update(nodes_new)
    for ip, pwd in nodes.items():
        c = Connection(ip, port=22, user='root', connect_kwargs={'password': pwd})
        c.run('rm -f /etc/hosts')
        c.put('/etc/hosts', '/etc/hosts')
        a = c.local('find /root/.ssh/ -name authorized_keys')
        if a.stdout.find('authorized_keys') != -1:
            c.run('rm -f /root/.ssh/authorized_keys')
        c.put('/root/.ssh/authorized_keys', '/root/.ssh/authorized_keys')
    print('over')


if __name__ == '__main__':
    os.makedirs('/opt/tsce4', exist_ok=True)
