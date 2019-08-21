import csv
import requests
import xml.etree.ElementTree as ET


def parseXML(xmlfile):

    # create element tree object
    # print(xmlfile)
    tree = ET.parse(xmlfile)

    # get root element
    root = tree.getroot()

    # create empty dictionary for errors
    # key: the path of the file, value: a list of errors(an error is represented by a dictionary)
    errorsFromManifest = {}

    # iterate news items
    for fileElement in root.findall('./testcase/file'):

        # list of errors from one file
        errorFromFile = []

        for child in fileElement:
            # empty news dictionary
            error = {}
            cweID = child.attrib['name'].split(': ')
            id=cweID[0].split('-')
            error['cwe'] = id[1]
            error['short description'] = cweID[1]
            error["line"] = child.attrib['line']
            error['true positive'] = '0'
            error['false positive'] = '0'
            error['false negative'] = '1'

            if error is not {}:
                errorFromFile.append(error)

        if errorFromFile:
            errorsFromManifest[fileElement.attrib['path']] = errorFromFile

    # return news items list
    return errorsFromManifest


def savetoCSV(finalDict, csvname):

    # specifying the fields for csv file
    fields = ['cwe', 'true positive', 'false positive',
              'false negative', 'path', 'line', 'short description']

    # writing to csv file
    # writing to csv file
    with open(csvname, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        for key in finalDict:
            errorList = finalDict[key]
            for error in errorList:
                row = [error['cwe'], error['true positive'], error['false positive'], error['false negative'], key,
                       error['line'], error['short description']]
                csvwriter.writerow(row)


def label(manifestDict, alertFile):
    # create element tree object
    # print(xmlfile)
    tree = ET.parse(alertFile)

    # get root element
    root = tree.getroot()

    # iterate through each flaw element in alertFile
    for errorElement in root.findall('./errors/error'):
        
        paths = []
        lines = []

        for location in errorElement:
            paths.append(location.attrib['file'])
            lines.append(location.attrib['line'])

        for index,path in enumerate(paths):
            #ffsa raised same alert in the same place the manifest mentioned
            if path in manifestDict:
                errorList = manifestDict[path]
                found = 0
                indexToBeUpdated =0
                for idx,error in enumerate(errorList):
                    key = 'cwe'
                    if key in errorElement.attrib and errorElement.attrib['cwe']==error['cwe'] and lines[index]==error['line']:
                        found = found + 1
                        indexToBeUpdated = idx
                        break
                
                if found == 0:
                    manifestDict[path][indexToBeUpdated]['false positive'] = '1'
                    manifestDict[path][indexToBeUpdated]['false negative'] = '0'
                else:
                    #print("here")
                    manifestDict[path][indexToBeUpdated]['true positive'] = '1'
                    manifestDict[path][indexToBeUpdated]['false negative'] = '0'
    
    return manifestDict
        


def main():

    # parse xml file
    itcList = parseXML('manifestITC.xml')
    # print(itcList)

    # label data by comapring dictionary made from manifest with alert file
    itcList = label(itcList, "itc.xml")

    #save the updated dictionary in csv
    savetoCSV(itcList, 'labeledITC.csv')


main()
