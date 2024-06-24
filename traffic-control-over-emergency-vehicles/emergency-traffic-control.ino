// Pin Definitions
int ledA1 = 2, ledA2 = 3, ledA3 = 4;
int ledB1 = 5, ledB2 = 6, ledB3 = 7;
int ledC1 = 8, ledC2 = 9, ledC3 = 10;
int ledD1 = 11, ledD2 = 12, ledD3 = 13;
int sensorA = A0, sensorB = A1, sensorC = A2, sensorD = A3;
int sirenPin = A4; // Analog pin to detect the siren of emergency vehicles

void setup() {
  pinMode(ledA1, OUTPUT); pinMode(ledA2, OUTPUT); pinMode(ledA3, OUTPUT);
  pinMode(ledB1, OUTPUT); pinMode(ledB2, OUTPUT); pinMode(ledB3, OUTPUT);
  pinMode(ledC1, OUTPUT); pinMode(ledC2, OUTPUT); pinMode(ledC3, OUTPUT);
  pinMode(ledD1, OUTPUT); pinMode(ledD2, OUTPUT); pinMode(ledD3, OUTPUT);
  pinMode(sensorA, INPUT); pinMode(sensorB, INPUT); pinMode(sensorC, INPUT); pinMode(sensorD, INPUT);
  pinMode(sirenPin, INPUT);
  Serial.begin(9600); // Initialize serial communication
}

void loop() {
  readSensor();
}

void readSensor() {
  int valA = analogRead(sensorA);
  int valB = analogRead(sensorB);
  int valC = analogRead(sensorC);
  int valD = analogRead(sensorD);
  bool emergency = digitalRead(sirenPin);
  
  if (Serial.available() > 0) {
    char emergencySignal = Serial.read();
    if (emergencySignal == 'A') {
      roadAopen();
    } else if (emergencySignal == 'B') {
      roadBopen();
    } else if (emergencySignal == 'C') {
      roadCopen();
    } else if (emergencySignal == 'D') {
      roadDopen();
    }
  } else {
    if (valA > valB && valA > valC && valA > valD) {
      roadAopen();
    } else if (valB > valA && valB > valC && valB > valD) {
      roadBopen();
    } else if (valC > valA && valC > valB && valC > valD) {
      roadCopen();
    } else {
      roadDopen();
    }
  }
}

void roadAopen() {
  digitalWrite(ledA3, LOW); digitalWrite(ledB3, HIGH); digitalWrite(ledC3, HIGH); digitalWrite(ledD3, HIGH);
  digitalWrite(ledA1, HIGH); delay(10000);
  digitalWrite(ledA1, LOW); digitalWrite(ledA2, HIGH); delay(1000);
  digitalWrite(ledA2, LOW); readSensor();
}

void roadBopen() {
  digitalWrite(ledB3, LOW); digitalWrite(ledA3, HIGH); digitalWrite(ledC3, HIGH); digitalWrite(ledD3, HIGH);
  digitalWrite(ledB1, HIGH); delay(10000);
  digitalWrite(ledB1, LOW); digitalWrite(ledB2, HIGH); delay(1000);
  digitalWrite(ledB2, LOW); readSensor();
}

void roadCopen() {
  digitalWrite(ledC3, LOW); digitalWrite(ledA3, HIGH); digitalWrite(ledB3, HIGH); digitalWrite(ledD3, HIGH);
  digitalWrite(ledC1, HIGH); delay(10000);
  digitalWrite(ledC1, LOW); digitalWrite(ledC2, HIGH); delay(1000);
  digitalWrite(ledC2, LOW); readSensor();
}

void roadDopen() {
  digitalWrite(ledD3, LOW); digitalWrite(ledA3, HIGH); digitalWrite(ledB3, HIGH); digitalWrite(ledC3, HIGH);
  digitalWrite(ledD1, HIGH); delay(10000);
  digitalWrite(ledD1, LOW); digitalWrite(ledD2, HIGH); delay(1000);
  digitalWrite(ledD2, LOW); readSensor();
}
