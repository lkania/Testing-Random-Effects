import hashlib


def hash_(string):
	return int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16)
