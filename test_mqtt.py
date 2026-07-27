from src.ingestion.mqtt_client import MQTTIngestor

def test_mqtt_ingestor_init():
    mqtt = MQTTIngestor(broker_host="localhost", broker_port=1883, topic="ev/test")
    assert mqtt.topic == "ev/test"
