
import time
import datetime
import os

import lakeapi
import plotly
import cufflinks

cufflinks.go_offline()
plotly.io.renderers.default = "colab"


# # =测试数据 ===
# lakeapi.use_sample_data(anonymous_access = True)

# books = lakeapi.load_data(
#     table="book",
#     start=datetime.datetime(2022, 10, 1),
#     end=datetime.datetime(2022, 10, 2),
#     symbols=["BTC-USDT"],
#     # columns=['receipt_time', 'bid_0_price', 'ask_0_price'],
#     exchanges=None,
# )
# # books.set_index('received_time', inplace = True)

# print(books)

# # # ===================
# 写一个方法,下载数据:
# 1, 传入symbols和start,end 
# 2,下载数据保存到 f'data/crypto-lake/book/BINANCE_FUTURES/{symbols}/{symbols}_{start}.feather'目录下


import lakeapi

def download_binance_futures_depth(symbols: str, start: datetime.datetime, end: datetime.datetime):
    """
    下载Binance合约深度数据并保存到指定目录
    
    Args:
        symbols: 交易对名称,如 'BTC-USDT-PERP'
        start: 开始时间
        end: 结束时间
    """
    # 创建保存目录
    save_dir = f'crypto-lake/book/OKX_FUTURES/{symbols}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{symbols}_{start.strftime("%Y%m%d")}.feather'
    # 检查文件是否已存在
    if os.path.exists(save_path):
        print(f"文件已存在，跳过下载: {save_path}")
        return

    # 下载数据
    df = lakeapi.load_data(
        table="book",
        start=start,
        end=end,
        symbols=[symbols],
        exchanges=["OKX"]
    )
    
    # 保存数据
    df.to_feather(save_path)
    print(f"数据已保存到: {save_path}")


if __name__ == '__main__':

    # 下载多个币种的数据
    symbols_list = [
            "BTC-USDT-PERP",
            "ETH-USDT-PERP"
                    ]  # 可以扩展更多币种
    start = datetime.datetime(2025, 1, 1)  # 起始日期
    end = datetime.datetime(2025, 4, 8)    # 结束日期
    start_dates = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
    print(f"将要下载的日期: {start_dates}")

    # 外层循环遍历所有币种
    for symbol in symbols_list:
        print(f"\n开始处理{symbol}的数据...")
        
        # 内层循环遍历日期
        for start_date in start_dates:
            # 计算结束时间(第二天的00:00:00)
            end_date = start_date + datetime.timedelta(days=1)
            
            print(f"开始下载 {symbol} {start_date.strftime('%Y-%m-%d')} 的数据...")
            
            try:
                download_binance_futures_depth(
                    symbols=symbol,
                    start=start_date,
                    end=end_date
                )
                print(f"{symbol} {start_date.strftime('%Y-%m-%d')} 数据下载完成")
                
            except Exception as e:
                print(f"下载 {symbol} {start_date.strftime('%Y-%m-%d')} 数据时发生错误: {str(e)}")
                
            # 添加短暂休息,避免请求过于频繁
            time.sleep(2)  # 增加休息时间，因为要处理更多数据
        
        print(f"{symbol}的所有日期数据下载完成!")
    
    print("\n所有币种的数据下载完成!")


