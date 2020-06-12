from dynsys.app import SimpleApp


class MyApp(SimpleApp):

    def __init__(self):
        super(MyApp, self).__init__("My App")


if __name__ == '__main__':
    MyApp().run()
