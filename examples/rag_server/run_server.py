from dotenv import load_dotenv
load_dotenv()

import uvicorn
import os
from .server import app

if __name__ == "__main__":
    
    port = os.getenv("PORT", 8080)
    port = int(port)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
