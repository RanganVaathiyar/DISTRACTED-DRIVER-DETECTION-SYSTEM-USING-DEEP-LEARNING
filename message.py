import pywhatkit 
 
 
 
# using Exception Handling to avoid 
# unprecedented errors 
try: 
 
 
 
# sending message to receiver 
# using pywhatkit 
pywhatkit.sendwhatmsg_instantly("+916383732118", 
 
"Hello driver! you are distracted, please drive safe :)",30) 
print("Successfully Sent!") 
 
 
except: 
 
 
 
# handling exception 
 
# and printing error message print("An Unexpected Error!") 
