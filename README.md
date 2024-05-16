## Getting Started

Welcome to the VS Code Java world. Here is a guideline to help you get started to write Java code in Visual Studio Code.

## Folder Structure

The workspace contains two folders by default, where:

- `src`: the folder to maintain sources
- `lib`: the folder to maintain dependencies

Meanwhile, the compiled output files will be generated in the `bin` folder by default.

> If you want to customize the folder structure, open `.vscode/settings.json` and update the related settings there.

## Dependency Management

The `JAVA PROJECTS` view allows you to manage your dependencies. More details can be found [here](https://github.com/microsoft/vscode-java-dependency#manage-dependencies).
## Preprocessing 
**Input**: IND,RAIN,IND.1,T.MAX,IND.2,T.MIN,T.MIN.G

**Output**: WIND
**Insignificant feature**: DATE
- Load data, read data : LoadData(), printTop10Data()
- Central Tendency of data: calculateMean()
- Remove Columns: removeLabel()
- Dispersion of data: calculateStandardDeviation(), calculatePercentile()
- Check duplicated, null : cleanNullValue()
- Remove outliers: removeOutliers()
- Featuring data: replaceStringByDouble()
