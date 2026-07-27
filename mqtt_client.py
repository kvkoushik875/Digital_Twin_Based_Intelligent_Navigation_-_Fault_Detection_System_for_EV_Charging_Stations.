import json
import threading
import time
from typing import Callable

import paho.mqtt.client as mqtt

class MQTTIngestor:
    def __init__(self, broker_host="localhost", broker_port=1883, topic="ev/charging/sensors"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client = mqtt.Client()
        self.latest_message = None
        self._callback: Callable[[dict], None] = None

    def _on_connect(self, client, userdata, flags, rc):
        print("MQTT connected with result code", rc)
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            self.latest_message = data
            if self._callback:
                self._callback(data)
        except Exception as e:
            print("Error parsing MQTT message:", e)

    def start(self, on_message: Callable[[dict], None] = None):
        self._callback = on_message
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker_host, self.broker_port, 60)

        thread = threading.Thread(target=self.client.loop_forever, daemon=True)
        thread.start()

    def get_latest(self):
        return self.latest_message
