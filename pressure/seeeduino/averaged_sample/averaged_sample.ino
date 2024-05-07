// References:
// - https://blog.thea.codes/getting-the-most-out-of-the-samd21-adc/
// - https://www.sigmdel.ca/michel/ha/xiao/seeeduino_xiao_01_en.html/

unsigned int ADCValue = 0;
float relaxation = 0.1;

void setup() {
  // put your setup code here, to run once:

  analogReadResolution(12);

  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  for(int i=0; i<100; i++){
  ADCValue = (1-relaxation)*ADCValue + relaxation*analogRead(A10);
  delay(1);
  }
  
  // print out the value you read:
  Serial.println(ADCValue);
  delay(1);        // delay in between reads for stability

}
