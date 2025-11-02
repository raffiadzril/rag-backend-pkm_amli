"""
Test Chatbot Service - Verify AI responses quality
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.chatbot_service import get_chatbot_service

def test_chatbot():
    """Test chatbot dengan berbagai pertanyaan"""
    
    print("=" * 80)
    print("🧪 TESTING CHATBOT AI QUALITY")
    print("=" * 80)
    print()
    
    # Initialize chatbot
    print("📚 Initializing chatbot service...")
    try:
        chatbot = get_chatbot_service()
        print("✅ Chatbot initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return
    
    # Test questions - dari mudah ke kompleks
    test_questions = [
        {
            "category": "📌 Prinsip Dasar",
            "question": "Apa saja prinsip MPASI?",
            "expected": "4 prinsip MPASI"
        },
        {
            "category": "👶 Usia & Waktu",
            "question": "Kapan bayi boleh mulai MPASI?",
            "expected": "6 bulan"
        },
        {
            "category": "🍽️ Menu & Bahan",
            "question": "Menu MPASI apa yang bagus untuk bayi 8 bulan?",
            "expected": "Contoh menu dengan bahan konkret"
        },
        {
            "category": "📏 Porsi",
            "question": "Berapa porsi MPASI untuk bayi 9 bulan?",
            "expected": "Jumlah ml atau sendok makan"
        },
        {
            "category": "⏰ Frekuensi",
            "question": "Berapa kali sehari bayi 7 bulan harus makan?",
            "expected": "Frekuensi makan dan snack"
        },
        {
            "category": "🥣 Tekstur",
            "question": "Tekstur MPASI seperti apa untuk bayi 6 bulan?",
            "expected": "Lumat halus/pure"
        },
        {
            "category": "🥩 Protein",
            "question": "Berapa kebutuhan protein untuk bayi 10 bulan?",
            "expected": "Angka gram protein"
        },
        {
            "category": "🚫 Makanan Terlarang",
            "question": "Makanan apa yang tidak boleh diberikan ke bayi?",
            "expected": "List makanan yang dilarang"
        },
        {
            "category": "⚠️ Alergi",
            "question": "Bayi saya alergi telur, pakai apa sebagai pengganti protein?",
            "expected": "Alternatif protein"
        },
        {
            "category": "🔧 Praktis",
            "question": "Bagaimana cara menyimpan MPASI yang benar?",
            "expected": "Tips penyimpanan"
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Test #{i} - {test['category']}")
        print(f"{'='*80}")
        print(f"\n❓ PERTANYAAN: {test['question']}")
        print(f"🎯 EXPECTED: {test['expected']}")
        print(f"\n{'.'*80}")
        
        try:
            # Get response from chatbot
            result = chatbot.generate_response(test['question'])
            
            if result['status'] == 'success':
                response = result['response']
                sources_used = result['sources_used']
                has_context = result['has_context']
                
                print(f"\n💬 JAWABAN AI:")
                print(f"{response}")
                print(f"\n{'.'*80}")
                print(f"📊 Stats:")
                print(f"  - Sources used: {sources_used}")
                print(f"  - Has context: {'Yes ✅' if has_context else 'No ❌'}")
                print(f"  - Response length: {len(response)} characters")
                print(f"  - Word count: {len(response.split())} words")
                
                # Simple quality check
                quality_score = 0
                if len(response) > 100:
                    quality_score += 1
                    print(f"  ✅ Sufficient length")
                else:
                    print(f"  ⚠️ Too short")
                
                if has_context and sources_used > 0:
                    quality_score += 1
                    print(f"  ✅ Used knowledge base")
                else:
                    print(f"  ⚠️ No knowledge base used")
                
                # Check for emoji
                if any(char in response for char in ['🍽️', '🥗', '👶', '✅', '❌', '⚠️', '📌', '💡']):
                    quality_score += 1
                    print(f"  ✅ Contains emoji")
                
                # Check for specific keywords based on question
                if test['category'] == "📌 Prinsip Dasar" and 'prinsip' in response.lower():
                    quality_score += 1
                elif test['category'] == "👶 Usia & Waktu" and ('6 bulan' in response.lower() or '6bulan' in response.lower()):
                    quality_score += 1
                elif test['category'] == "🍽️ Menu & Bahan" and any(food in response.lower() for food in ['beras', 'ayam', 'wortel', 'sayur', 'daging']):
                    quality_score += 1
                elif test['category'] == "📏 Porsi" and any(unit in response.lower() for unit in ['ml', 'sendok', 'mangkok', 'gram']):
                    quality_score += 1
                elif test['category'] == "⏰ Frekuensi" and any(freq in response.lower() for freq in ['kali', 'sehari', 'pagi', 'siang', 'malam']):
                    quality_score += 1
                elif test['category'] == "🥣 Tekstur" and any(tex in response.lower() for tex in ['lumat', 'halus', 'lembut', 'kental', 'pure']):
                    quality_score += 1
                elif test['category'] == "🥩 Protein" and any(num in response for num in ['gram', 'g']):
                    quality_score += 1
                elif test['category'] == "🚫 Makanan Terlarang" and any(food in response.lower() for food in ['madu', 'gula', 'garam', 'susu sapi']):
                    quality_score += 1
                elif test['category'] == "⚠️ Alergi" and any(alt in response.lower() for alt in ['ikan', 'ayam', 'daging', 'tahu', 'tempe']):
                    quality_score += 1
                elif test['category'] == "🔧 Praktis":
                    quality_score += 1
                
                if quality_score >= 3:
                    print(f"\n✅ QUALITY: GOOD ({quality_score}/4)")
                    results.append(("PASS", test['question']))
                else:
                    print(f"\n⚠️ QUALITY: NEEDS IMPROVEMENT ({quality_score}/4)")
                    results.append(("FAIL", test['question']))
                    
            else:
                print(f"\n❌ ERROR: {result['response']}")
                results.append(("ERROR", test['question']))
                
        except Exception as e:
            print(f"\n❌ EXCEPTION: {str(e)}")
            results.append(("ERROR", test['question']))
        
        print()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    pass_count = sum(1 for r in results if r[0] == "PASS")
    fail_count = sum(1 for r in results if r[0] == "FAIL")
    error_count = sum(1 for r in results if r[0] == "ERROR")
    
    print(f"\n✅ PASSED: {pass_count}/{len(results)}")
    print(f"⚠️ FAILED: {fail_count}/{len(results)}")
    print(f"❌ ERRORS: {error_count}/{len(results)}")
    
    if pass_count >= len(results) * 0.8:
        print(f"\n🎉 OVERALL: EXCELLENT - AI is responding well!")
    elif pass_count >= len(results) * 0.6:
        print(f"\n👍 OVERALL: GOOD - AI is responding adequately")
    else:
        print(f"\n⚠️ OVERALL: NEEDS IMPROVEMENT - Check knowledge base and prompts")
    
    print("\n" + "="*80)
    print()


if __name__ == "__main__":
    test_chatbot()
