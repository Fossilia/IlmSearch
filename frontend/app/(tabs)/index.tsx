import React, { useState } from 'react';
import { 
  View, 
  Text, 
  Keyboard, 
  Alert, 
  FlatList, 
  TouchableOpacity 
} from 'react-native';
import { Searchbar, ActivityIndicator, Card, Divider } from 'react-native-paper'; 

import styles from './styles'; 
import { SearchResult, QuranResult, HadithResult, search } from '../../services/searchService';
import * as Clipboard from 'expo-clipboard';
import Feather from '@expo/vector-icons/Feather';

export default function SearchScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult | null>(null);
  const [hasSearched, setHasSearched] = useState(false); // Controls the layout shift
  const [activeTab, setActiveTab] = useState<'quran' | 'hadith'>('quran');

   const copyToClipboard = async (text: string) => {
    await Clipboard.setStringAsync(text);
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    Keyboard.dismiss();
    setLoading(true);
    setHasSearched(true); // Triggers the "move up" animation effect
    setActiveTab('quran'); 

    try {
      const data = await search(searchQuery, 3);
      setResults(data);
      
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'Failed to fetch results. Check your connection.');
    } finally {
      setLoading(false);
    }
  };

  const renderQuranItem = ({ item }: { item: QuranResult }) => {
    // Handle error/missing content case
    if (item.error) {
      return (
        <Card style={[styles.card, styles.errorCard]}>
          <Card.Content>
            <Text style={styles.errorText}>⚠️ {item.reference}: {item.error}</Text>
          </Card.Content>
        </Card>
      );
    }

    // Regular Verse Card
    return (
      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.cardHeader}>
            <Text style={styles.surahName}>{item.surah_name}</Text>
            <Text style={styles.verseRef}>{item.reference}</Text>
          </View>
          
          <Divider style={styles.divider} />
          
          <Text style={styles.arabicText}>{item.arabic}</Text>
          <Text style={styles.englishText}>{item.english}</Text>

          <Divider style={styles.divider} />
          <TouchableOpacity 
          onPress={() => copyToClipboard(`${item.arabic}\n\n${item.english}`)}>
            <Feather name="copy" size={24} color="black" />
          </TouchableOpacity>

        </Card.Content>
      </Card>
    );
  };

  const renderHadithItem = ({ item }: { item: HadithResult }) => {
    if (item.error) {
      return (
        <Card style={[styles.card, styles.errorCard]}>
          <Card.Content>
            <Text style={styles.errorText}>⚠️ {item.id || 'Unknown'}: {item.error}</Text>
          </Card.Content>
        </Card>
      );
    }

    return (
      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.cardHeader}>
            <Text style={styles.surahName}>{item.book}</Text>
            <Text style={styles.verseRef}>{item.id?.split(":")[1]}</Text>
          </View>
          
          <Divider style={styles.divider} />
          
          <Text style={styles.englishText}>{item.english}</Text>
          <Text style={styles.arabicText}>{item.arabic}</Text>
          <Divider style={styles.divider} />
          <TouchableOpacity 
            onPress={() => copyToClipboard(`${item.arabic}\n\n${item.english}`)}>
            <Feather name="copy" size={24} color="black" />
          </TouchableOpacity>
        </Card.Content>
      </Card>
    );
  };

  return (
    <View style={[
      styles.container, 
      hasSearched ? { justifyContent: 'flex-start', paddingTop: 60 } : { justifyContent: 'center' }
    ]}>
      
      {!hasSearched && (
        <View style={{ alignItems: 'center' }}>
          <Text style={[styles.title, { color: 'black' }]}>Ilm Search</Text>
          <Text style={[styles.subtitle, { color: 'gray' }]}>Enter a topic or question</Text>
        </View>
      )}

      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search topics..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          onSubmitEditing={handleSearch}
          style={styles.searchBar}
          loading={loading} 
        />
      </View>

      {hasSearched && (
        <View style={styles.resultsArea}>
          
          {/* Custom Tab Selector */}
          <View style={styles.tabContainer}>
            <TouchableOpacity 
              style={[styles.tab, activeTab === 'quran' && styles.activeTab]}
              onPress={() => setActiveTab('quran')}
            >
              <Text style={[styles.tabText, activeTab === 'quran' && styles.activeTabText]}>Qur'an</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.tab, activeTab === 'hadith' && styles.activeTab]}
              onPress={() => setActiveTab('hadith')}
            >
              <Text style={[styles.tabText, activeTab === 'hadith' && styles.activeTabText]}>Hadith</Text>
            </TouchableOpacity>
          </View>

          {/* CONTENT AREA */}
          {loading ? (
             <ActivityIndicator style={{ marginTop: 50 }} size="large" />
          ) : (
            <>
              {activeTab === 'quran' ? (
                <FlatList
                  data={results?.quran || []}
                  keyExtractor={(item, index) => item.reference + index}
                  renderItem={renderQuranItem}
                  contentContainerStyle={styles.listContent}
                  showsVerticalScrollIndicator={false}
                />
              ) : (
                <FlatList
                  data={results?.hadith || []}
                  keyExtractor={(item, index) => (item.id || 'unknown') + index}
                  renderItem={renderHadithItem}
                  contentContainerStyle={styles.listContent}
                  showsVerticalScrollIndicator={false}
                />
              )}
            </>
          )}
        </View>
      )}
    </View>
  );
};