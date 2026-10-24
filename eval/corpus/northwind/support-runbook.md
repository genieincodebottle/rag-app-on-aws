# Field Support Runbook

## Error E-401, calibration drift

E-401 means the measured tool position differs from the commanded position by more than the tolerance. Stop the program, run the calibration routine from the teach pendant menu Service > Calibrate, then run the test pattern. If E-401 returns within 24 hours, open a ticket with the serial number and the last 50 log lines.

## Error E-502, motor over-temperature

E-502 means a joint motor exceeded 85 degrees Celsius. Let the arm cool for at least 30 minutes with the controller powered on so the fans keep running. Check that the cabinet air filter is clean. Repeated E-502 errors need an engineer visit.

## Error E-118, emergency stop circuit open

E-118 means the emergency stop chain is open. Check every emergency stop button and the safety fence door switch, then reset the controller.

## Escalation

Customer-down incidents are escalated to the on-call engineer through the support hotline. The response target for a customer-down incident is four hours.
