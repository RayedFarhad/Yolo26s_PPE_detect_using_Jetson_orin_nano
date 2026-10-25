/*
  Vest gate controller with feedback to the Jetson

  Jetson -> Arduino (9600 baud, one per line):
    "GO"      -> open servo
    "STOP"    -> close servo
    "UNKNOWN" -> keep servo closed
  The Jetson resends its current command every second (keep-alive).

  Arduino -> Jetson (one per line, simple "KEY:VALUE" format):
    READY            board has started
    ACK:GO           command received (also ACK:STOP, ACK:UNKNOWN)
    GATE:OPEN        LDR says the gate is physically open
    GATE:CLOSED      LDR says the gate is physically closed
                     (sent on every change AND every 500 ms as a heartbeat)
    EVENT:TIMEOUT    no command from the Jetson for 3 s -> gate closed
    ERR:<text>       unknown command received
    LDR:<value>      raw LDR reading (only when DEBUG_LDR = true)

  LEDs: GREEN = gate open, RED = gate closed (from the LDR, not the command)

  Wiring:
    Servo signal -> D9   (servo power from 5 V supply, common GND)
    LDR          -> A0   (LDR from 5 V to A0, 10k resistor from A0 to GND)
    Green LED    -> D11  (through a 220 ohm resistor)
    Red LED      -> D12  (through a 220 ohm resistor)
*/

#include <Servo.h>

// ---------- Pins ----------
const int SERVO_PIN     = 9;
const int LDR_PIN       = A0;
const int GREEN_LED_PIN = 11;
const int RED_LED_PIN   = 12;

// ---------- Servo angles ----------
const int SERVO_OPEN_ANGLE   = 180;
const int SERVO_CLOSED_ANGLE = 85;

// ---------- LDR ----------
const bool LDR_OPEN_WHEN_BRIGHT = true;
const int  LDR_THRESHOLD  = 605;   // midpoint of 450 (covered) and 760 (uncovered)
const int  LDR_HYSTERESIS = 30;
const bool DEBUG_LDR      = false;

// ---------- Timing ----------
const unsigned long COMMAND_TIMEOUT_MS = 3000;  // close gate if Jetson goes silent
const unsigned long HEARTBEAT_MS       = 500;   // report gate state this often

Servo gateServo;
String receivedData = "";
bool servoOpen = false;
bool gateOpenLdr = false;
bool timedOut = false;
unsigned long lastCommandMs = 0;
unsigned long lastHeartbeatMs = 0;
unsigned long lastDebugMs = 0;

void setServo(bool open) {
  if (open == servoOpen) return;
  servoOpen = open;
  gateServo.write(open ? SERVO_OPEN_ANGLE : SERVO_CLOSED_ANGLE);
}

void sendGateState() {
  Serial.println(gateOpenLdr ? "GATE:OPEN" : "GATE:CLOSED");
}

void handleCommand(const String &cmd) {
  lastCommandMs = millis();
  timedOut = false;

  if (cmd == "GO") {
    setServo(true);
  }
  else if (cmd == "STOP" || cmd == "UNKNOWN") {
    setServo(false);
  }
  else {
    Serial.print("ERR:");
    Serial.println(cmd);
    return;
  }

  Serial.print("ACK:");
  Serial.println(cmd);
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      receivedData.trim();
      if (receivedData.length() > 0) {
        handleCommand(receivedData);
      }
      receivedData = "";
    }
    else if (receivedData.length() < 15) {
      receivedData += c;
    }
    else {
      receivedData = "";
    }
  }
}

void showGateLeds(bool open) {
  digitalWrite(GREEN_LED_PIN, open ? HIGH : LOW);
  digitalWrite(RED_LED_PIN,   open ? LOW  : HIGH);
}

void updateLdr() {
  int value = analogRead(LDR_PIN);
  bool bright = gateOpenLdr == LDR_OPEN_WHEN_BRIGHT
                  ? value > LDR_THRESHOLD - LDR_HYSTERESIS
                  : value > LDR_THRESHOLD + LDR_HYSTERESIS;
  bool open = LDR_OPEN_WHEN_BRIGHT ? bright : !bright;

  if (open != gateOpenLdr) {
    gateOpenLdr = open;
    showGateLeds(open);
    sendGateState();                       // report every change immediately
  }

  if (DEBUG_LDR && millis() - lastDebugMs >= 500) {
    lastDebugMs = millis();
    Serial.print("LDR:");
    Serial.println(value);
  }
}

void sendHeartbeat() {
  if (millis() - lastHeartbeatMs >= HEARTBEAT_MS) {
    lastHeartbeatMs = millis();
    sendGateState();                       // lets the Jetson know we are alive
  }
}

void checkTimeout() {
  if (!timedOut && millis() - lastCommandMs > COMMAND_TIMEOUT_MS) {
    timedOut = true;
    setServo(false);
    Serial.println("EVENT:TIMEOUT");
  }
}

void setup() {
  Serial.begin(9600);
  receivedData.reserve(16);

  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  showGateLeds(false);

  gateServo.attach(SERVO_PIN);
  gateServo.write(SERVO_CLOSED_ANGLE);
  servoOpen = false;
  lastCommandMs = millis();

  Serial.println("READY");
}

void loop() {
  readSerial();
  updateLdr();
  sendHeartbeat();
  checkTimeout();
}
