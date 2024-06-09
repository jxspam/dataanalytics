# Data Dictionary and Data Miscellaneous

| Column Name            | Description                                                                                       |
|------------------------|---------------------------------------------------------------------------------------------------|
| ID                     | Trip identification number                                                                       |
| VendorID               | A code indicating the TPEP provider that provided the record. 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc. |
| tpep_pickup_datetime  | The date and time when the meter was engaged.                                                    |
| tpep_dropoff_datetime | The date and time when the meter was disengaged.                                                 |
| Passenger_count        | The number of passengers in the vehicle. This is a driver-entered value.                          |
| Trip_distance         | The elapsed trip distance in miles reported by the taximeter.                                     |
| PULocationID          | TLC Taxi Zone in which the taximeter was engaged.                                                 |
| DOLocationID          | TLC Taxi Zone in which the taximeter was disengaged.                                              |
| RateCodeID            | The final rate code in effect at the end of the trip. 1= Standard rate; 2= JFK; 3= Newark; 4= Nassau or Westchester; 5= Negotiated fare; 6= Group ride  |
| Store_and_fwd_flag    | This flag indicates whether the trip record was held in vehicle memory before being sent to the vendor, aka “store and forward,” because the vehicle did not have a connection to the server. Y= store and forward trip; N= not a store and forward trip |
| Payment_type           | A numeric code signifying how the passenger paid for the trip. 1= Credit card; 2= Cash; 3= No charge; 4= Dispute; 5= Unknown; 6= Voided trip |
| Fare_amount            | The time-and-distance fare calculated by the meter.                                               |
| Extra                  | Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges. |
| MTA_tax                | $0.50 MTA tax that is automatically triggered based on the metered rate in use.                 |
| Improvement_surcharge  | $0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015. |
| Tip_amount             | Tip amount – This field is automatically populated for credit card tips. Cash tips are not included. |
| Tolls_amount           | Total amount of all tolls paid in trip.                                                          |
| Total_amount           | The total amount charged to passengers. Does not include cash tips.                               |
---

## More Information (Answereed by ChatGPT):

Q: What is the RateCode ID implying?

1. ___Standard rate___: This rate applies to regular taxi trips within the standard operating area.

2. ___JFK___: This rate applies specifically to trips going to or from John F. Kennedy International Airport (JFK) in New York City.

3. ___Newark___: This rate applies specifically to trips going to or from Newark Liberty International Airport in Newark, New Jersey.

4. ___Nassau or Westchester___: This rate applies to trips going to or from locations in Nassau County or Westchester County, which are neighboring areas outside of New York City.

5. ___Negotiated fare___: This rate applies to trips where the fare has been negotiated between the passenger and the driver, often for longer-distance or specialized trips.

6. ___Group ride___: This rate applies to trips where multiple passengers share the same taxi ride, often reducing the cost for each individual passenger.