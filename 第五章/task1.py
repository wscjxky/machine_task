import requests
import threading


def f():
    req=requests.post('http://567cg.cn/Default/Update',
                  {'phonecode': 123123,
    'action': '提交短信验证码'})
    print(req.text)
    req = requests.post('http://567cg.cn/Default/Update',
                        {
                         'action': '获取验证码'})
    print(req.text)

f()
threadlist = []
for i in range(10):
    threadlist.append(threading.Thread(target=f))
for t in threadlist:
    t.setDaemon(True)  # 如果你在for循环里用，不行， 因为上一个多线程还没结束又开始下一个
    t.start()
for j in threadlist:
    j.join()