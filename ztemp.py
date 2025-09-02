from temporal.our_exp import *

cfg = Config.from_custom_yaml_file(r"./config/base.yaml")
current_mt = MethodFactory.create_method(cfg)
pprint(current_mt)
