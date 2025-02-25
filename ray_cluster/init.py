import ray

context = ray.init(address="ray://35.94.226.73:10001")
print(context.dashboard_url)