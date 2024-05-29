import redis

# Create a connection to the Redis server
r = redis.Redis(host="localhost", port=6379, db=0)

# Set a key-value pair
r.set("mykey", "myvalue")

# Retrieve the value of the key
value = r.get("mykey")
print(value.decode("utf-8"))  # Outputs: myvalue

# Set multiple key-value pairs
r.mset({"key1": "value1", "key2": "value2"})

# Get multiple values using their keys
values = r.mget(["key1", "key2"])
for val in values:
    print(val.decode("utf-8"))

# Delete a key-value pair
r.delete("mykey")

# Check if a key exists
exists = r.exists("mykey")
print(exists)  # Outputs: 0
