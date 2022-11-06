const byte DIR = 2;
const byte PUL = 3;
const float pasos = 200;
const float gvuelta = 1.8;
const int velocidad = 300;
const int grados = 9;
void girar_derecha();

void setup() {
    pinMode(DIR, OUTPUT);
    pinMode(PUL, OUTPUT);
}

void loop() {
  int i = 0;
  gradosr = i*gvuelta;
  while(gradosr <= grados){
    girar_derecha();
    gradosr = i*gvuelta;  
  }
}

void girar_derecha()
{
    digitalWrite(DIR, LOW);
    digitalWrite(PUL, HIGH);
    delayMicroseconds(velocidad);
    digitalWrite(PUL, LOW);
    delayMicroseconds(velocidad);
}
