import json
from wsgiref.simple_server import make_server
 
 
# 定义函数，参数是函数的两个参数，都是python本身定义的，默认就行了。
def application(environ, start_response):
    # 定义文件请求的类型和当前请求成功的code
    start_response('200 OK', [('Content-Type', 'application/json')])
    # environ是当前请求的所有数据，包括Header和URL，body
 
    request_body = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH", 0)))
    request_body = json.loads(request_body.decode("utf-8"))
 
    name = request_body["name"]
    no = request_body["no"]
 
    dic = {'myNameIs': name, 'myNoIs': no}
 
    return [json.dumps(dic)]
 
 
if __name__ == "__main__":
    port = 6088
    httpd = make_server("127.0.0.1", port, application)
    print("serving http on port {0}...".format(str(port)))
    httpd.serve_forever()
