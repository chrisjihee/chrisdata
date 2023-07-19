import ipaddress
import json

import httpx
import netifaces

from chrisbase.io import JobTimer


def available_ipv4_addresses():
    for interface in netifaces.interfaces():
        if netifaces.AF_INET in netifaces.ifaddresses(interface):
            for address_info in netifaces.ifaddresses(interface)[netifaces.AF_INET]:
                address_object = ipaddress.IPv4Address(address_info['addr'])
                if address_object.is_global:
                    yield address_info['addr']


ips = sorted(list(available_ipv4_addresses()))


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
