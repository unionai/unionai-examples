_sales_trends_query = """
SELECT 
    year_09_10.StockCode,
    year_09_10.Description,
    AVG(year_09_10.Quantity) AS Avg_Quantity_Historical,
    AVG(Recent.Quantity) AS Avg_Quantity_Recent
FROM 
    e_commerce.year_09_10
LEFT JOIN 
    mydf AS Recent
ON 
    year_09_10.StockCode = Recent.StockCode
WHERE 
    year_09_10.Quantity > 0 AND Recent.Quantity > 0
GROUP BY 
    year_09_10.StockCode, year_09_10.Description
HAVING 
    AVG(Recent.Quantity) IS NOT NULL
ORDER BY 
    (AVG(Recent.Quantity) - AVG(year_09_10.Quantity)) DESC
"""

sales_trends_query = """
WITH HistoricalData AS (
    SELECT 
        StockCode,
        Description,
        AVG(Quantity) AS Avg_Quantity_Historical
    FROM 
        e_commerce.year_09_10
    WHERE 
        Quantity > 0
    GROUP BY 
        StockCode, Description
),
RecentData AS (
    SELECT 
        StockCode,
        AVG(Quantity) AS Avg_Quantity_Recent
    FROM 
        mydf
    WHERE 
        Quantity > 0
    GROUP BY 
        StockCode
)
SELECT 
    HistoricalData.StockCode,
    HistoricalData.Description,
    HistoricalData.Avg_Quantity_Historical,
    RecentData.Avg_Quantity_Recent
FROM 
    HistoricalData
LEFT JOIN 
    RecentData 
ON 
    HistoricalData.StockCode = RecentData.StockCode
WHERE 
    RecentData.Avg_Quantity_Recent IS NOT NULL
ORDER BY 
    (RecentData.Avg_Quantity_Recent - HistoricalData.Avg_Quantity_Historical) DESC
"""

_elasticity_query = """
SELECT 
    year_09_10.StockCode,
    year_09_10.Description,
    CORR(Recent.Price, Recent.Quantity) AS Recent_Price_Elasticity,
    CORR(year_09_10.Price, year_09_10.Quantity) AS Historical_Price_Elasticity,
    ABS(CORR(Recent.Price, Recent.Quantity) - CORR(year_09_10.Price, year_09_10.Quantity)) AS Elasticity_Change
FROM 
    e_commerce.year_09_10 
LEFT JOIN 
    mydf AS Recent
ON 
    year_09_10.StockCode = Recent.StockCode
WHERE 
    year_09_10.Quantity > 0 AND Recent.Quantity > 0
GROUP BY 
    year_09_10.StockCode, year_09_10.Description
HAVING 
    CORR(Recent.Price, Recent.Quantity) IS NOT NULL 
    AND CORR(year_09_10.Price, year_09_10.Quantity) IS NOT NULL
ORDER BY 
    Elasticity_Change DESC;
"""

elasticity_query = """
WITH HistoricalData AS (
    SELECT 
        StockCode,
        Description,
        CORR(Price, Quantity) AS Historical_Price_Elasticity
    FROM 
        e_commerce.year_09_10
    WHERE 
        Quantity > 0
    GROUP BY 
        StockCode, Description
),
RecentData AS (
    SELECT 
        StockCode,
        CORR(Price, Quantity) AS Recent_Price_Elasticity
    FROM 
        mydf
    WHERE 
        Quantity > 0
    GROUP BY 
        StockCode
)
SELECT 
    HistoricalData.StockCode,
    HistoricalData.Description,
    RecentData.Recent_Price_Elasticity,
    HistoricalData.Historical_Price_Elasticity,
    ABS(RecentData.Recent_Price_Elasticity - HistoricalData.Historical_Price_Elasticity) AS Elasticity_Change
FROM 
    HistoricalData
LEFT JOIN 
    RecentData 
ON 
    HistoricalData.StockCode = RecentData.StockCode
WHERE 
    RecentData.Recent_Price_Elasticity IS NOT NULL
    AND HistoricalData.Historical_Price_Elasticity IS NOT NULL
ORDER BY 
    Elasticity_Change DESC;
"""

_outlier_query = """
SELECT 
    year_09_10.StockCode,
    year_09_10.Description,
    AVG(year_09_10.Quantity) AS Avg_Historical_Quantity,
    AVG(Recent.Quantity) AS Avg_Recent_Quantity,
    AVG(Recent.Quantity) - AVG(year_09_10.Quantity) AS Deviation
FROM 
    e_commerce.year_09_10  
LEFT JOIN 
    mydf AS Recent
ON 
    year_09_10.StockCode = Recent.StockCode
WHERE 
    year_09_10.Quantity > 0 AND Recent.Quantity > 0 AND year_09_10.Description IS NOT NULL AND Recent.Description IS NOT NULL
GROUP BY 
    year_09_10.StockCode, year_09_10.Description
ORDER BY 
    Deviation DESC;
"""

outlier_query = """
WITH HistoricalData AS (
    SELECT 
        StockCode,
        Description,
        AVG(Quantity) AS Avg_Historical_Quantity
    FROM 
        e_commerce.year_09_10
    WHERE 
        Quantity > 0 AND Description IS NOT NULL
    GROUP BY 
        StockCode, Description
),
RecentData AS (
    SELECT 
        StockCode,
        AVG(Quantity) AS Avg_Recent_Quantity
    FROM 
        mydf
    WHERE 
        Quantity > 0 AND Description IS NOT NULL
    GROUP BY 
        StockCode
)
SELECT 
    HistoricalData.StockCode,
    HistoricalData.Description,
    HistoricalData.Avg_Historical_Quantity,
    RecentData.Avg_Recent_Quantity,
    RecentData.Avg_Recent_Quantity - HistoricalData.Avg_Historical_Quantity AS Deviation
FROM 
    HistoricalData
LEFT JOIN 
    RecentData 
ON 
    HistoricalData.StockCode = RecentData.StockCode
ORDER BY 
    Deviation DESC;
"""

customer_segmentation_query = """
WITH HistoricalData AS (
    SELECT 
        "Customer ID",
        COUNT(DISTINCT Invoice) AS Historical_Transactions,
        SUM(Quantity * Price) AS Historical_Spend
    FROM 
        e_commerce.year_09_10
    WHERE 
        Quantity > 0 AND Quantity > 0
    GROUP BY 
        "Customer ID"
),
RecentData AS (
    SELECT 
        "Customer ID",
        COUNT(DISTINCT Invoice) AS Recent_Transactions,
        SUM(Quantity * Price) AS Recent_Spend
    FROM 
        mydf
    WHERE 
        Quantity > 0 AND Quantity > 0
    GROUP BY 
        "Customer ID"
)
SELECT 
    HistoricalData."Customer ID",
    RecentData.Recent_Transactions,
    HistoricalData.Historical_Transactions,
    100.0 * RecentData.Recent_Transactions / NULLIF(HistoricalData.Historical_Transactions, 0) AS Transaction_Percentage,
    RecentData.Recent_Spend,
    HistoricalData.Historical_Spend,
    100.0 * RecentData.Recent_Spend / NULLIF(HistoricalData.Historical_Spend, 0) AS Spend_Percentage
FROM 
    HistoricalData
LEFT JOIN 
    RecentData 
ON 
    HistoricalData."Customer ID" = RecentData."Customer ID"
ORDER BY 
    Spend_Percentage DESC
"""