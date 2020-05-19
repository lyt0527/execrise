# -*- encoding: utf-8 -*-
import paramiko
import time
import json

def get_connection(cube_name, end_time):
    # 创建SSH对象
    ssh = paramiko.SSHClient()
    # 允许连接不在know_hosts文件中的主机
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 连接服务器
    ssh.connect(hostname='96.4.0.10',
                port=22,
                username='root',
                password='1Z26P18vXuqu8kP0',
                timeout=10)

    uuid = get_start(ssh, cube_name, end_time)
    res_len_err = get_status(ssh, uuid)
    ssh.close()
    return res_len_err

def get_status(ssh, uuid):
    print("=============")
    status_command = 'curl -s -X GET --user ADMIN:KYLIN -H "Content-Type: application/json;charset=utf-8" http://172.22.14.119:7070/kylin/api/jobs/' + uuid
    status_command = status_command.strip('\r\n')
    stdin_status, stdout_status, stderr_status = ssh.exec_command(status_command)
    result_status = stdout_status.readlines()
    result_status = json.loads(result_status[0])['job_status']
    print(result_status,'2222222222')
    return result_status

def get_start(ssh, cube_name, end_time):
    print(end_time,ssh)
    # st = '{"startTime": %s, "endTime":%s, "buildType":"BUILD"}' % ('"' + start_time + '"', '"' + end_time + '"')
    st = '{"endTime":%s, "buildType":"BUILD"}' % ('"' + end_time + '"')
    start_command = 'curl -s -X PUT --user ADMIN:KYLIN -H "Content-Type: application/json;charset=utf-8" -d ' + "'" + st + "'"\
    +' http://172.22.14.119:7070/kylin/api/cubes/{}/build'.format(cube_name)
    status_command = 'curl -s -X GET --user ADMIN:KYLIN -H "Content-Type: application/json;charset=utf-8" http://172.22.14.119:7070/kylin/api/jobs/4c71e1cd-c78b-cadd-409d-cc45e0008b32'

    start_command = start_command.strip('\r\n')
    stdin_start, stdout_start, stderr_start = ssh.exec_command(start_command)
    result_start = stdout_start.readlines()
    print(result_start)
    uuid = json.loads(result_start[0])['uuid']
    print(uuid,'111111111')
    return uuid

if __name__ == '__main__':
    # now_time = datetime.datetime.now()
    # end_time = now_time - datetime.timedelta(hours=now_time.hour, minutes=now_time.minute, seconds=now_time.second, microseconds=now_time.microsecond)
    # start_time = (end_time+datetime.timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S")
    # end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    # start_time = '2019-12-23 00:00:00'
    end_time = '2019-12-26 00:01:00'
    end_time = str(int(time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")))*1000)
    cube_names = ['test1', 'test2']
    for cube_name in cube_names:
        re = get_connection(cube_name, end_time)
    print(re)
    print('finished')


