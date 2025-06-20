#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.python_executor import execute_python_code

async def test_calculation():
    # Test with proper print statement (what the AI should generate now)
    args = {
        'code': 'print((3+2+1+1231231+2139018230912)/3+120/99+2012)'
    }
    print('Testing with proper print statement:', repr(args['code']))
    
    result = await execute_python_code(args)
    print('Success:', result.get('success', False))
    print('Output:', repr(result.get('output', '')))
    print('Expected result: 713006489396.2122')
    
    # Test another calculation
    args2 = {
        'code': '''
result = (3+2+1+1231231+2139018230912)/3+120/99+2012
print(f"The calculation result is: {result}")
'''
    }
    print('\nTesting with formatted output:', repr(args2['code']))
    
    result2 = await execute_python_code(args2)
    print('Success:', result2.get('success', False))
    print('Output:', repr(result2.get('output', '')))

if __name__ == "__main__":
    asyncio.run(test_calculation())
