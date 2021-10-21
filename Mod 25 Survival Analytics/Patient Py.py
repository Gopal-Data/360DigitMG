import pandas as pd
from lifelines import KaplanMeierFitter  
# Loading the Patient Dataset 
patient = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 252\\Patient.csv")
patient.head()
patient.describe()

#PatientID : Name (Nominal Data not very important)
#Followup : Time
#Eventtype : Event
#Scenario: Group (Everyone are from same group)

T = patient.Followup 
E = patient.Eventtype

#Fitting KaplanMeierFitter model 
kmf = KaplanMeierFitter() 
kmf.fit( T, event_observed= E) 

# Time-line estimations plot 
kmf.plot()