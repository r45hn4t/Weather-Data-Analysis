import numpy as np;import pandas as pd;import matplotlib.pyplot as plt
csvData = pd.read_csv('clever_weather_data.csv',usecols=['Sensor ID', 'Temperature', 'Message Time Stamp'])
csvData.head(3)
def cleanup(csvData):
    fil = (csvData['Temperature'] >= 0.0) & (csvData['Temperature'] <= 45.0)
    csvData = csvData[fil]
    node_data = [x[-1] for x in csvData['Sensor ID'].str.split("_")]
    date_data = [x[0] for x in csvData['Message Time Stamp'].str.split("T")]
    time_data = [x[1] for x in csvData['Message Time Stamp'].str.split("T")]
    date_data = pd.Series(pd.to_datetime(date_data))
    csvData["node"]=node_data
    csvData["date"]=date_data
    csvData["time"]=[x[:8] for x in time_data]
    csvData['node'] = pd.to_numeric(csvData['node'])
    csvData.drop(['Sensor ID'],axis=1,inplace=True)
    csvData.drop(['Message Time Stamp'],axis=1,inplace=True)
    return csvData
csvData = cleanup(csvData)
csvData.head()

plt.figure(figsize=(18,9))
mask1  = (csvData["node"]==1) & (csvData["date"]=="2021-02-01")
mask2 = (csvData["node"]==2) & (csvData["date"]=="2021-02-01")
node001 = csvData[mask1]
node002 = csvData[mask2]
node002 = node002.sort_values("time")
plt1=plt.plot(node001["time"],node001["Temperature"])[0]
plt2 = plt.plot(node002["time"],node002["Temperature"])[0]
plt.legend(["node 001","node 002"])
plt.title("Temparature vs Time in node 001 and node 002",fontsize=20)
plt.xlabel("Time",fontsize=14)
plt.ylabel("Temparature",fontsize=14)
plt.xticks(ticks=[node002["time"].iloc[4*x]for x in range(0,13)],fontsize=9)
plt.show()
try:
    import pandas as pd

    csvData = pd.DataFrame(
        pd.read_csv('clever_weather_data.csv'),
        columns=['Sensor ID', 'Temperature', 'Message Time Stamp'])
    pd.set_option('display.max_columns', 4)

    # task_1
    fil = (csvData['Temperature'] >= 0.0) & (csvData['Temperature'] <= 45.0)

    csvData = csvData.loc[fil]

    node_data = list(csvData['Sensor ID'])
    node_data = list(set([int(x.strip().split('_')[-1]) for x in node_data]))

    date_data = list(csvData['Message Time Stamp'])
    date_data = list(set([x.strip().split('T')[0] for x in date_data]))


    def task_2():
        menu = """
        
        1. Find out the consecutive time slots when temperature was below a certain temperature reported by a node.
        2. Find all nodes which have reported temperature in a given range of temperature.
        3. Find out the maximum and minimum temperature for all nodes in a day. Also display the difference of temperatures on each node.
        4. Find out for a given day, announce a fire warning for each node that reports temperature above 33 C.
        5. Find out for a given month, find the maximum variation in temperature.
        6. Find out for a given day, average temperature for each node, average temperature (rounded values) equal to temperature given,  sorted distinct average temperatures.
        0. Exit
        
        """

        while True:
            print(menu)
            x = input("Enter an option: ")

            if x == '1':
                date = input("Enter a Date[YYYY-MM-DD]: ")
                node = input("Enter a Node number: ")
                temp = input("Enter a Temperature: ")
                task_3(date, node, temp)
                print("")
            elif x == '2':
                date = input("Enter a Date[YYYY-MM-DD]: ")
                temp1 = input("Enter a Temperature 1: ")
                temp2 = input("Enter a Temperature 2: ")
                task_4(date, temp1, temp2)
                print("")
            elif x == '3':
                date = input("Enter a Date[YYYY-MM-DD]: ")
                task_5(date)
                print("")
            elif x == '4':
                date = input("Enter a Date[YYYY-MM-DD]: ")
                task_6(date, '33')
                print("")
            elif x == '5':
                month = input("Enter a Month[MM]: ")
                task_7(month)
                print("")
            elif x == '6':
                date = input("Enter a Date[YYYY-MM-DD]: ")
                r = task_8(date)
                print("")
                temp = input("Enter a Temperature: ")
                task_8_2(r, temp)
                print("")
                task_8_3(r)
                print("")
            elif x == '0':
                break
            else:
                print("Please choose valid option!")


    def task_3(date, node, temp):
        if date not in date_data:
            print("No Such Date Exists in Data")
            return

        if not (node.isdigit() or temp.isdigit()):
            print("Temperature and node number should be number not alphabet")
            return

        node = int(node)
        temp = float(node)

        if node not in node_data:
            print("No data is available for given node")
            return

        return_data_list = []
        slot_data = ""
        flag = 0
        for row in csvData.itertuples():
            node_temp = row[1].strip().split('_')[-1]
            node_number = int(node_temp)

            if node_number == node:
                date_only = row[3].strip().split('T')[0]
                time_only = row[3].strip().split('T')[1].strip().split('.')

                if date_only == date:
                    #if row[2] < temp:
                    if flag == 0:
                        slot_data += time_only[0]
                        flag = 1
                    else:
                        slot_data += "-" + time_only[0]
                        return_data_list.append(slot_data)
                        slot_data = ""
                        flag = 0

        if len(return_data_list) > 0:
            print("Time slots when temperature was below {} C: ".format(temp))
            print(return_data_list)
            return
        print("No data is available for given day")


    def task_4(date, temp1, temp2):
        if date not in date_data:
            print("No Such Date Exists")
            return
        if not (temp1.isdigit() or temp2.isdigit()):
            print("Temperature should be number not alphabet")
            return

        min_t = min(float(temp1), float(temp2))
        max_t = max(float(temp1), float(temp2))

        return_data_list = []

        for row in csvData.itertuples():
            date_only = row[3].strip().split('T')[0]

            if date_only == date:
                if min_t <= row[2] <= max_t:
                    node = row[1].strip().split('_')[-1]

                    if node not in return_data_list:
                        return_data_list.append(node)

        return_data_list = sorted(return_data_list)
        print("Nodes which have reported temperature in range of {} C to {} C : ".format(min_t, max_t))
        print(return_data_list)


    def task_5(date):
        node_list_of_day = []
        return_data_dict = {}
        return_data_dict_temp_diff = {}
        index_list = []

        if date not in date_data:
            print("No Such Date Exists")
            return
        for row in csvData.itertuples():
            date_temp = row[3].split('T')
            date_only = date_temp[0].strip()

            if date_only == date:
                node_temp = row[1].strip().split('_')
                if node_temp[-1] not in node_list_of_day:
                    node_list_of_day.append(node_temp[-1])

                index_list.append(row[0])

        sr = csvData.loc[index_list, ['Sensor ID', 'Temperature', 'Message Time Stamp']]

        if not len(node_list_of_day) > 0:
            print("No data found for the given date")
            return

        for node in node_list_of_day:
            min_t = 46.0
            max_t = -1.0
            for row in sr.itertuples():
                node_temp = row[1].strip().split('_')
                node_number = node_temp[-1]

                if node_number == node:
                    if row[2] < min_t:
                        min_t = row[2]
                    if row[2] > max_t:
                        max_t = row[2]

            return_data_dict.update({"node_" + node: str(min_t) + "-" + str(max_t)})
            return_data_dict_temp_diff.update({"node_" + node: str(round(max_t - min_t, 1))})

        # print("Temperature Difference in {}: ".format(date))
        # print(return_data_dict)
        # print(return_data_dict_temp_diff)

        return [return_data_dict,return_data_dict_temp_diff]


    def task_6(date, temp):
        node_list_of_day = []
        index_list = []

        if date not in date_data:
            print("No Such Date Exists")
            return

        if not (temp.isdigit()):
            print("Temperature should be number not alphabet")
            return

        for row in csvData.itertuples():
            date_temp = row[3].split('T')
            date_only = date_temp[0].strip()

            if date_only == date:
                node_temp = row[1].strip().split('_')
                if node_temp[-1] not in node_list_of_day:
                    node_list_of_day.append(node_temp[-1])
                index_list.append(row[0])

        sr = csvData.loc[index_list, ['Sensor ID', 'Temperature', 'Message Time Stamp']]

        if not len(node_list_of_day) > 0:
            print("No data found for the given date")
            return

        count = 0
        for node in node_list_of_day:
            flag = 0
            for row in sr.itertuples():
                node_temp = row[1].strip().split('_')
                node_number = node_temp[-1]

                if node_number == node:
                    if row[2] > float(temp):
                        print("Be careful,high fire risk around {} with temperature {} C".format("node_" + node, row[2]))
                        flag = 1
            if flag == 1:
                count += 1

        print("Total Number of Nodes in Fire Warning: ", count)


    def task_7(month):
        months_data = list(set([x.split('-')[1] for x in date_data]))
        if month not in months_data:
            print("Month doesn't exists in the data")
            return

        min_t = 46.0
        max_t = -1.0
        #min_t = max_t = csvData["Temperature"][0]
        min_date_time_index = 0
        max_date_time_index = 0
        for row in csvData.itertuples():
            date_only = row[3].strip().split('T')[0]
            month_only = date_only.split('-')[1]
            if month_only == month:
                if row[2] < min_t:
                    min_t = row[2]
                    min_date_time_index = row[0]
                if row[2] > max_t:
                    max_t = row[2]
                    max_date_time_index = row[0]

        sr = list(csvData.loc[[min_date_time_index, max_date_time_index], 'Message Time Stamp'])
        result = []
        for t in sr:
            date = t.strip().split('T')[0]
            time = t.strip().split('T')[1].split('.')[0]
            result.append(date + "::" + time)
        result.append(max_t - min_t)
        print("Maximum Variation for the Month NO.{} : ".format(month))
        print(result)
        if result[2] > 15:
            print("Warning! temperature difference is more than 15C")


    def task_8(date):
        return_data_dict = {}
        if date not in date_data:
            print("No Such Date Exists")
            return
        for row in csvData.itertuples():
            s = row[3].strip().split('T')[0]
            if s == date:
                node = "node_" + str(row[1].strip().split('_')[-1])
                if node not in return_data_dict.keys():
                    return_data_dict.update({node: (row[2], 1)})
                else:
                    return_data_dict[node] = (return_data_dict[node][0] + row[2], return_data_dict[node][1] + 1)

        for k, v in return_data_dict.items():
            return_data_dict[k] = round(return_data_dict[k][0] / return_data_dict[k][1], 2)

        print("Average temperature for each node in {}: ".format(date))
        print(return_data_dict)
        return return_data_dict


    def task_8_2(dic, temp):
        return_data_list = []
        if not (temp.isdigit()):
            print("Temperature should be number not alphabet")
            return
        for k, v in dic.items():
            if round(v) == int(temp):
                return_data_list.append(k)

        print("Nodes having average temperature of {}: ".format(temp))
        print(return_data_list)


    def task_8_3(dic):
        avg_temp_list = []
        for k, v in dic.items():
            avg_temp_list.append(v)

        avg_temp_list = list(set(avg_temp_list))
        n = len(avg_temp_list)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if avg_temp_list[j] > avg_temp_list[j + 1]:
                    avg_temp_list[j], avg_temp_list[j + 1] = avg_temp_list[j + 1], avg_temp_list[j]
        print("Sorted distinct average temperatures:")
        print(avg_temp_list)

except:
    pass
temp_dict = task_5("2021-02-08")[0]
temp_dict_diff = task_5("2021-02-08")[1]
temp_df = pd.DataFrame()
temp_df["node"]= temp_dict_diff.keys()
temp_df["temp"]= temp_dict_diff.values()
temp_df = temp_df.sort_values("node")
temp_df = temp_df.reset_index()
temp_df.drop("index",axis=1,inplace=True)
temp_df.temp = pd.to_numeric(temp_df.temp)
plt.figure(figsize=(20,10))
plt.plot(temp_df["node"],temp_df["temp"])
plt.xlabel("Node",fontsize=14);plt.ylabel("Temparature",fontsize=14);plt.title("Temaparature of nodes in 2021-02-08",fontsize=20)
plt.xticks(temp_df.node[::3])
plt.figure(figsize=(20,10))
plt.bar(temp_df["node"],temp_df["temp"])
plt.xlabel("Node",fontsize=14);plt.ylabel("Temparature",fontsize=14);plt.title("Temaparature of nodes in 2021-02-08",fontsize=20)
plt.xticks(temp_df.node[::3])
plt.show()
dict  = task_8("2021-01-13")
df = pd.DataFrame()
df["node"]=dict.keys()
df["temp"]=dict.values()
df["node_no"]=[int(x[1]) for x in df["node"].str.split("_")]
df = df.sort_values(["node_no"])
df.set_index("node_no")
plt.figure(figsize=(20,6))
plt1 = plt.subplot(1,2,1);plt2 = plt.subplot(1,2,2)
plt1.plot(df["node"],df["temp"],"r-")
plt2.scatter(df["node"],df["temp"])
plt1.legend(["Average Temparature"],fontsize=11,loc="upper left")
plt2.legend(["Average Temparature"],fontsize=11,loc="upper left")
plt1.set_title("Nodewise Temparature (Line plot)",fontdict={"fontsize":"20"}) 
plt2.set_title("Nodewise Temparature (Scatter plot)",fontdict={"fontsize":"20"}) 
# set("Nodewise Temparature",fontdict={"fontsize":"22"})
plt1.set_xticks(np.arange(1,len(df.node),5))
plt2.set_xticks(np.arange(1,len(df.node),5))
plt1.set_xlabel("Nodes",fontdict={"fontsize":"15"});plt2.set_xlabel("Nodes",fontdict={"fontsize":"15"})
plt1.set_ylabel("Temparature",fontdict={"fontsize":"15"});plt2.set_ylabel("Temparature",fontdict={"fontsize":"15"})
plt.show()
csvData = cleanup(csvData)
temperature = 23
date="2021-02-08"
def node_selector(date,temperature):
    nodes  = csvData["node"].unique()
    selected_nodes=[]
    for node in nodes:
        mask1 = (csvData["node"]==node)
        mask2 = (csvData["Temperature"]>temperature)
        mask3 = (csvData["date"]==date)
        mask = (mask1)&(mask2)&(mask3)
        if mask.sum()>=10:
            selected_nodes.append(node)
    if len(selected_nodes)>15:
        selected_nodes=selected_nodes[:15]
    else:
        print("Not enough nodes")    
    return selected_nodes
selected_nodes = node_selector(date,temperature)
plt.figure(figsize=(20,9))
for node in selected_nodes:
    mask = (csvData.node==node)
    temps = csvData[mask]["Temperature"][:10]
    times0 = csvData[mask]["time"][:10]
    times = pd.Series(times0)
    times = pd.to_timedelta(times)
    times.sort_values()
    plt.plot(times,temps)
plt.legend([f"node {node}" for node in selected_nodes])
plt.xlabel("Time of the Day",fontsize=14)
plt.ylabel("Time of the Day",fontsize=14)
plt.title("Temparature on selected Nodes",fontsize=20)
plt.xticks([])
plt.show()
plt.figure(figsize=(20,9))
plt.figure(figsize=(20,9))
for node in selected_nodes:
    mask = (csvData.node==node)
    temps = csvData[mask]["Temperature"][:10]
    times0 = csvData[mask]["time"][:10]
    times = pd.Series(times0)
    times = pd.to_timedelta(times)
    times.sort_values()
    plt.plot(times,temps,linewidth=7)
plt.legend([f"node {node}" for node in selected_nodes])
plt.xlabel("Time of the Day",fontsize=14)
plt.ylabel("Time of the Day",fontsize=14)
plt.title("Temparature on selected Nodes",fontsize=20)
plt.xticks([])
plt.grid(axis="both")
plt.show()




csvData.time = pd.to_timedelta(csvData.time)
csvData.date = pd.to_datetime(csvData.date)
sel_df=csvData.set_index("date")
sel_df = sel_df[pd.to_datetime("2021-01-01"):pd.to_datetime("2021-01-04")]
sel_df = sel_df.reset_index()
dates = sel_df.date.unique()
fig, axs = plt.subplots(2,2,figsize=(18,18))
date="2021-01-01"
sel_df = sel_df[sel_df.date==date]
templist=[]
labels=["00:00:01-04:00:00","04:00:01-08:00:00","08:00:01-12:00:00","12:00:01-16:00:00","16:00:01-20:00:00","20:00:01-00:00:00"]
templist.append(sel_df[(sel_df.time<pd.to_timedelta("04:00:01")) & (sel_df.time>pd.to_timedelta("00:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("08:00:00")) & (sel_df.time>pd.to_timedelta("04:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("12:00:00")) & (sel_df.time>pd.to_timedelta("08:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("16:00:00")) & (sel_df.time>pd.to_timedelta("12:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("20:00:00")) & (sel_df.time>pd.to_timedelta("16:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time>pd.to_timedelta("20:00:01"))]["Temperature"].mean())
templist.sort(reverse=True)
axs[0,0].pie(templist,explode=[0.1,0,0,0,0,0],labels=labels,autopct="%1.1f")
axs[0,0].set_title("2021-01-01",fontsize=17)
csvData.time = pd.to_timedelta(csvData.time)
csvData.date = pd.to_datetime(csvData.date)
sel_df=csvData.set_index("date")
sel_df = sel_df[pd.to_datetime("2021-01-01"):pd.to_datetime("2021-01-04")]
sel_df = sel_df.reset_index()
dates = sel_df.date.unique()
date="2021-01-02"
sel_df = sel_df[sel_df.date==date]
templist=[]
labels=["00:00:01-04:00:00","04:00:01-08:00:00","08:00:01-12:00:00","12:00:01-16:00:00","16:00:01-20:00:00","20:00:01-00:00:00"]
templist.append(sel_df[(sel_df.time<pd.to_timedelta("04:00:01")) & (sel_df.time>pd.to_timedelta("00:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("08:00:00")) & (sel_df.time>pd.to_timedelta("04:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("12:00:00")) & (sel_df.time>pd.to_timedelta("08:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("16:00:00")) & (sel_df.time>pd.to_timedelta("12:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("20:00:00")) & (sel_df.time>pd.to_timedelta("16:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time>pd.to_timedelta("20:00:01"))]["Temperature"].mean())
templist.sort(reverse=True)
axs[0,1].pie(templist,explode=[0.1,0,0,0,0,0],labels=labels,autopct="%1.1f")
axs[0,1].set_title("2021-01-02",fontsize=17)
csvData.time = pd.to_timedelta(csvData.time)
csvData.date = pd.to_datetime(csvData.date)
sel_df=csvData.set_index("date")
sel_df = sel_df[pd.to_datetime("2021-01-01"):pd.to_datetime("2021-01-04")]
sel_df = sel_df.reset_index()
dates = sel_df.date.unique()
date="2021-01-03"
sel_df = sel_df[sel_df.date==date]
templist=[]
labels=["00:00:01-04:00:00","04:00:01-08:00:00","08:00:01-12:00:00","12:00:01-16:00:00","16:00:01-20:00:00","20:00:01-00:00:00"]
templist.append(sel_df[(sel_df.time<pd.to_timedelta("04:00:01")) & (sel_df.time>pd.to_timedelta("00:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("08:00:00")) & (sel_df.time>pd.to_timedelta("04:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("12:00:00")) & (sel_df.time>pd.to_timedelta("08:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("16:00:00")) & (sel_df.time>pd.to_timedelta("12:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("20:00:00")) & (sel_df.time>pd.to_timedelta("16:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time>pd.to_timedelta("20:00:01"))]["Temperature"].mean())
templist.sort(reverse=True)
axs[1,0].pie(templist,explode=[0.1,0,0,0,0,0],labels=labels,autopct="%1.1f")
axs[1,0].set_title("2021-01-03",fontsize=17)
csvData.time = pd.to_timedelta(csvData.time)
csvData.date = pd.to_datetime(csvData.date)
sel_df=csvData.set_index("date")
sel_df = sel_df[pd.to_datetime("2021-01-01"):pd.to_datetime("2021-01-04")]
sel_df = sel_df.reset_index()
dates = sel_df.date.unique()
date="2021-01-04"
sel_df = sel_df[sel_df.date==date]
templist=[]
labels=["00:00:01-04:00:00","04:00:01-08:00:00","08:00:01-12:00:00","12:00:01-16:00:00","16:00:01-20:00:00","20:00:01-00:00:00"]
templist.append(sel_df[(sel_df.time<pd.to_timedelta("04:00:01")) & (sel_df.time>pd.to_timedelta("00:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("08:00:00")) & (sel_df.time>pd.to_timedelta("04:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("12:00:00")) & (sel_df.time>pd.to_timedelta("08:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("16:00:00")) & (sel_df.time>pd.to_timedelta("12:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time<pd.to_timedelta("20:00:00")) & (sel_df.time>pd.to_timedelta("16:00:01"))]["Temperature"].mean())
templist.append(sel_df[(sel_df.time>pd.to_timedelta("20:00:01"))]["Temperature"].mean())
templist.sort(reverse=True)
axs[1,1].pie(templist,explode=[0.1,0,0,0,0,0],labels=labels,autopct="%1.1f")
axs[1,1].set_title("2021-01-04",fontsize=17)
plt.subplots_adjust(left=0.2,
                    bottom=0.1, 
                    right=1.1, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()