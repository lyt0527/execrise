import time
# import socket

# 获取本机IP
# hostname = socket.gethostname()
# ip = socket.gethostbyname(hostname)
# machine_id = int(ip.split('.')[3])
# print(machine_id)

class SnowFlake:
    def __init__(self,dataID):
        # 初始时间戳，毫秒级别
        self.start = 1585634723984
        self.last = int(round(time.time() * 1000))
        self.countID = 0
        self.dataID = dataID 

    def get_id(self):
        # 获取时间差
        now = int(round(time.time() * 1000))
        temp = now - self.start
        print(temp)
        # 时间差不够9位的在前面补0
        if len(str(temp)) < 9:
            length = len(str(temp))
            s = "0" * (9 - length)
            temp = s + str(temp)
        if now == self.last:
            self.countID += 1   # 同一时间差，序列号自增
        else:
            self.countID = 0    # 不同时间差，序列号重新置为0
            self.last = now
        # 标识ID部分
        if len(str(self.dataID)) < 2:
            length = len(str(self.dataID))
            s = "0" * (2 - length)
            self.dataID = s + str(self.dataID)
        # 自增序列号部分
        if self.countID == 99999:  # 序列号自增5位满了，睡眠一秒钟
            time.sleep(1)
        countIDdata = str(self.countID)
        if len(countIDdata) < 5:  # 序列号不够5位的在前面补0
            length = len(countIDdata)
            s = "0" * (5-length)
            countIDdata = s + countIDdata
        id = str(temp) + str(self.dataID) + countIDdata
        return id

if __name__ == '__main__':
    snow = SnowFlake(dataID = "00")
    print(len(snow.get_id()))
