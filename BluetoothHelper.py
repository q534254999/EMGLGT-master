import time
import asyncio
from collections import deque
from multiprocessing import Queue

import threading
from bleak import discover
from bleak import BleakClient

from bleak.backends.dotnet.client import BleakClientDotNet
import struct
SAMPLEFZ = 10
SAMPRATE = 10
DATASIZE = SAMPLEFZ * 60
PLOTSIZE = SAMPLEFZ * 9


def bytearr2int(bytearray, len):
    return struct.unpack('>'+'H'*int(len/2), bytearray)[0]


def callback(sender, data):

    # data[i] 是 10进制
    print("接收数据长度:")
    print(len(data))
    print("-------接收数据:-------")

    if len(data) > 1:
        print(data)
        # 查询设备状态及参数
        # print('-------解析参数:-------')
        # data[0] 指令类型
        if hex(data[0]) == '0x8b':
            print("-------参数查询返回:-------")
            if hex(data[2]) == '0xff':
                print('正在治疗')
            else:
                print('未启动治疗')
            if hex(data[3]) == '0x86':
                print('训练模式')
            print('治疗时间', bytearr2int(data[4:6], len(data[4:6])))
        # elif hex(data[0]) == '0x86':
        #     print("通道:", bytearr2int(data[2], len(data[2])))

        elif hex(data[0]) == '0x80':
            if hex(data[2]) == '0xff':
                print('开始治疗')
            else:
                print('未启动治疗')

        elif hex(data[0]) == '0x81':
            print('暂停后重新开始治疗')

        elif hex(data[0]) == '0x88':
            print('治疗暂停')

        elif hex(data[0]) == '0x90':
            if hex(data[3]) == '0x1':
                print(str(data[2])+'通道错误', "开路报警")

        elif hex(data[0]) == '0x91':
            if hex(data[3]) == '0x1':
                print(str(data[2])+'通道错误', "参数错误")
            elif hex(data[3]) == '0x2':
                print(str(data[2])+'通道错误', "校验错误")

        elif hex(data[0]) == '0x66':
            if len(data) == 5:
                print('肌电信号值：')
                print(int(hex(data[3])+hex(data[2])[-2:], 16))
                helper.dataQueue.append(int(hex(data[3])+hex(data[2])[-2:], 16))


class BluetoothHelper(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # 消息队列
        self.queue = Queue(maxsize=100)
        # 数据
        self.dataQueue = deque(maxlen=20 * DATASIZE)
        # 初次使用标志
        self.startFlag = 0
        self.Flag = True

        # 蓝牙配置
        self.device_name = "LGT-233"
        self.address = "CC:41:48:AA:B4:9A"

        # 特征UUID
        self.UUID_WRITE = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
        self.UUID_READ = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
        self.UUID_Name = "00002a29-0000-1000-8000-00805f9b34fb"
        self.UUID_HV = "00002A27-0000-1000-8000-00805f9b34fb"
        self.UUID_SV = "00002A28-0000-1000-8000-00805f9b34fb"
        self.UUID_BATTERY = "00002A19-0000-1000-8000-00805f9b34fb"

        # 控制指令
        # 查询设备指令
        self.DEVICE_STATUS = bytearray.fromhex('8B01008C')  # 0
        # 开启通道1 电刺激
        self.HEAL_PARA_1 = bytearray.fromhex('860C01500000010005050005100205')  # 1
        self.HEAL_PARA_2 = bytearray.fromhex('860C02500000010005050005100206')  # 2
        # 通道1 暂停
        self.PAUSECMD = bytearray.fromhex('88010089')  # 3
        # 通道1 重新开始电刺激
        self.RESTARTCMD = bytearray.fromhex('81010082')  # 4
        # 设置电刺激频率
        self.fre_cmd = bytearray.fromhex('8E0301050097')  # 5
        # 设置电刺激脉宽
        self.bandwidth_cmd = bytearray.fromhex('8E0302000194')  # 6
        # 设置电刺激强度
        self.value_cmd = bytearray.fromhex('8A03015000DE')  # 7

        # 控制指令序号
        self.DEVICE_STATUS_NO = 0
        self.HEAL_PARA_1_NO = 1
        self.HEAL_PARA_2_NO = 2
        self.PAUSECMD_NO = 3
        self.RESTARTCMD_NO = 4
        self.FRE_CMD_NO = 5
        self.BANDWIDTH_CMD_NO = 6
        self.VALUE_CMD_NO = 7

        self.bluetoothValue = 10


    # def start(self):
    #     loop = asyncio.new_event_loop()
    #     loop.run_until_complete(self.run(loop))


    def setHealFre(self, fre):
        fre = fre//10
        if fre <= 0:
            print("频率过小")
            return
        elif fre >= 100:
            print('频率过大')
            return

        self.fre_cmd = self.replacePara(self.fre_cmd, 3, fre)
        self.fre_cmd = self.replacePara(self.fre_cmd, 5, self.calArrCheckSum(self.fre_cmd))

    def setHealBandwith(self, bandwidth):
        bandwidth = bandwidth//100
        if bandwidth >= 100:
            print("带宽过大")
            return
        elif bandwidth <= 0:
            print("带宽过小")
            return

        self.bandwidth_cmd = self.replacePara(self.bandwidth_cmd, 4, bandwidth)
        self.bandwidth_cmd = self.replacePara(self.bandwidth_cmd, 5, self.calArrCheckSum(self.bandwidth_cmd))

    def setHealValue(self, value):
        if value >= 100:
            print("刺激强度过大")
            return
        self.bluetoothValue = value
        self.value_cmd = self.replacePara(self.value_cmd, 3, value)
        self.value_cmd = self.replacePara(self.value_cmd, 5, self.calArrCheckSum(self.value_cmd))
        self.sendCMD(self.VALUE_CMD_NO)

    # 替换发送数据中的参数
    def replacePara(self, bytearr, pos, para_dec):
        # 会自动转16进制
        bytearr[pos] = para_dec
        return bytearr

    # 计算结尾校验和
    def calArrCheckSum(self, bytearr):
        print(type(bytearr))
        checksum_dec = 0
        for i in range(len(bytearr)-1):
            checksum_dec += bytearr[i]
        # 返回的值是十进制
        return checksum_dec % 256

    # 向蓝牙发送指令
    def sendCMD(self, cmdID):
        if cmdID == self.HEAL_PARA_1_NO:
            if self.startFlag == 0:
                self.startFlag = 1
                print("--------通道1开启--------")
                self.queue.put(self.HEAL_PARA_1)
            else:
                print("--------电刺激重新开始--------")
                self.queue.put(self.RESTARTCMD)
        elif cmdID == self.HEAL_PARA_2_NO:
            print("--------通道2开启--------")
            self.queue.put(self.HEAL_PARA_2)
        elif cmdID == self.PAUSECMD_NO:
            print("--------电刺激暂停--------")
            self.queue.put(self.PAUSECMD)
        elif cmdID == self.RESTARTCMD_NO:
            if self.startFlag == 1:
                print("--------电刺激重新开始--------")
                self.queue.put(self.RESTARTCMD)
        elif cmdID == self.FRE_CMD_NO:
            print('--------修改电刺激频率--------')
            self.queue.put(self.fre_cmd)
        elif cmdID == self.BANDWIDTH_CMD_NO:
            print('--------修改电刺激脉宽--------')
            self.queue.put(self.bandwidth_cmd)
        elif cmdID == self.VALUE_CMD_NO:
            print('--------修改电刺激强度--------')
            self.queue.put(self.value_cmd)

    def stop(self):
        self.Flag = False

    def run(self):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.open(loop))

    async def open(self, loop):
        devices = await discover()
        for d in devices:
            print(d.name)
            if d.name == self.device_name:
                self.address = d.address

        print('try to connect to BLE 4.0')
        async with BleakClientDotNet(self.address, loop=loop) as client:
            x = client.is_connected
            print("Bluetooth Connected: {0}".format(x))
            await client.start_notify(self.UUID_READ, callback)
            try:
                while self.Flag:
                    # 读取肌电信号
                    await client.start_notify(self.UUID_READ, callback)
                    print(self.dataQueue)
                    # 取消息（指令）队列中的指令
                    if not self.queue.empty():
                        cmd = self.queue.get()
                        await client.write_gatt_char(self.UUID_WRITE, cmd, True)

                    # else:
                    #     print('no command in message queue')
                    time.sleep(0.5)
            except Exception as e:
                print("Exception", e)

            await client.stop_notify(self.UUID_READ)
            await client.disconnect()

            # Test
            # Notify
            # try:
            #     await client.start_notify(self.UUID_READ, callback)
            #     # 查询设备状态
            #     # print("发送数据:")
            #     # print(self.DEVICE_STATUS)
            #     # y = await client.write_gatt_char(self.UUID_WRITE, self.DEVICE_STATUS, True)
            #     # 设置参数开始治疗
            #     print("----------发送数据:----------")
            #     print(self.HEAL_PARA_1)
            #     await client.write_gatt_char(self.UUID_WRITE, self.HEAL_PARA_1, True)
            #     # 设置输出强度
            #     print("----------发送数据:----------")
            #     print(self.value_cmd)
            #     y = await client.write_gatt_char(self.UUID_WRITE, self.value_cmd, True)
            #     z = await client.read_gatt_char(self.UUID_READ)
            #     print("输出强度返回:", z)
            #     print("----------发送数据:----------")
            #     print(self.PAUSECMD)
            #     y = await client.write_gatt_char(self.UUID_WRITE, self.PAUSECMD, True)
            #     z = await client.read_gatt_char(self.UUID_READ)
            #     print("暂停返回:", z)
            #     print("----------发送数据:----------")
            #     print(self.RESTARTCMD)
            #     y = await client.write_gatt_char(self.UUID_WRITE, self.RESTARTCMD, True)
            #     z = await client.read_gatt_char(self.UUID_READ)
            #     print("重开返回:", z)
            #     # print("----------发送数据:----------")
            #     # print(self.HEAL_PARA_2)
            #     # y = await client.write_gatt_char(UUID_WRITE, self.HEAL_PARA_2, True)
            #     # z = await client.read_gatt_char(UUID_READ)
            #     # print("接收数据长度:")
            #     # print(len(z))
            #     # print("接收数据")
            #     # print(z)
            #     # for i in range(len(z)):
            #     #     print(z[i])
            #     # print(y)
            #     # y_9 = await client.read_gatt_char(UUID9)
            #     # print("Manufacturer Name:")
            #     # print(str(y_9))
            #     # y_7 = await client.read_gatt_char(UUID7)
            #     # print("Hardware Revision:")
            #     #
            #     # print(y_7)
            #     # y_8 = await client.read_gatt_char(UUID8)
            #     # print("Software Revision:")
            #     # print(y_8)
            #     # y_battery = await client.read_gatt_char(self.UUID_BATTERY)
            #     # print("Battery Level:")
            #     # print(y_battery[0])
            #
            #     # await asyncio.sleep(10.0)
            #
            #     await client.stop_notify(self.UUID_READ)
            #     await client.disconnect()
            # except Exception as e:
            #     print(e)


if __name__ == "__main__":
    helper = BluetoothHelper()
    helper.start()
    # while True:
    #     print(helper.dataQueue)
    #     time.sleep(1)




