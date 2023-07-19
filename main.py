from ipaddress import IPv4Address
import json

import httpx
import netifaces

from chrisbase.io import JobTimer


def local_ip_addrs():
    for inf in netifaces.interfaces():
        inf_addrs = netifaces.ifaddresses(inf).get(netifaces.AF_INET)
        if inf_addrs:
            for inf_addr in [x.get('addr') for x in inf_addrs]:
                if inf_addr and IPv4Address(inf_addr).is_global:
                    yield inf_addr


ips = sorted(list(local_ip_addrs()))


def testInetAddr(title="* Check all IP addresses"):
    with JobTimer(verbose=True, name=title, rt=1, rb=1):
        for ip in ips:
            with httpx.Client(transport=httpx.HTTPTransport(local_address=ip)) as client:
                r = client.get("https://api64.ipify.org?format=json", timeout=10.0)
                res = json.loads(r.text)
                src = '.'.join(ip.rsplit('.', maxsplit=2)[1:])
                info = ' ---- '.join(map(lambda x: f"[{x}]", [f"{src:<7s}", f"{r.status_code}",
                                                              f"{r.elapsed.total_seconds() * 1000:7,.0f}ms", f"{r.num_bytes_downloaded / 1024:7,.2f}KB"]))
                print(f"  - checked result: {res} ---- {info}")


if __name__ == "__main__":
    print(ips)
