import threading


def run_with_timeout(func, timeout: float) -> bool:
    """在 daemon 线程中执行 func，最多等待 timeout 秒。

    func 在 worker 线程中运行，应通过闭包写共享状态（典型是含
    output/error 字段的 result dict）。返回 True 表示超时（worker 仍在运行）。
    """
    t = threading.Thread(target=func, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return t.is_alive()