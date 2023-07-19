from chrisbase.net import ips, check_ip_addrs

if __name__ == "__main__":
    print(f"IPs = {', '.join(ips)}")
    check_ip_addrs()
