import React from 'react';
import { View, ScrollView, Text, Image, StyleSheet } from 'react-native';

const Info = () => {
    return (
        <ScrollView style={styles.container}>
            <View style={styles.section}>
                <Text style={styles.title}>Introduction to Climate Change and Power Grid</Text>
                <Text style={styles.content}>
                Climate change refers to long-term changes in temperature and weather patterns. The power grid is a complex
                network of electrical components that supply electricity from producers to consumers.
                </Text>
            </View>
            <View style={styles.section}>
                <Text style={styles.title}>Impact of Climate Change on the Power Grid</Text>
                <Text style={styles.content}>
                Rising temperatures increase electricity demand, especially for cooling. Extreme weather events can damage
                infrastructure, causing power outages and disruptions.
                </Text>
                <Image source={{ uri: 'url_to_relevant_image' }} style={styles.image} />
            </View>
            <View style={styles.section}>
                <Text style={styles.title}>Data and Insights</Text>
                <Text style={styles.content}>
                Power usage increases during heatwaves, straining the grid. Renewable energy sources like solar and wind are
                essential for reducing carbon emissions.
                </Text>
            </View>
            <View style={styles.section}>
                <Text style={styles.title}>Optimization and Technology Solutions</Text>
                <Text style={styles.content}>
                Optimization involves making the best use of available resources. Quantum algorithms and machine learning can
                improve grid management, reduce energy waste, and enhance reliability.
                </Text>
            </View>
            <View style={styles.section}>
                <Text style={styles.title}>Future Trends and Innovations</Text>
                <Text style={styles.content}>
                Smart grids, decentralized energy systems, and emerging technologies offer promising solutions for a more
                sustainable energy future.
                </Text>
            </View>
            <View style={styles.section}>
                <Text style={styles.title}>Resources and Further Reading</Text>
                <Text style={styles.content}>
                Explore additional resources and join organizations advocating for sustainable energy practices.
                </Text>
            </View>
        </ScrollView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
        padding: 10,
    },
    section: {
        marginBottom: 20,
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 5,
    },
    content: {
        fontSize: 16,
        lineHeight: 24,
        color: '#333',
    },
    image: {
        width: '100%',
        height: 200,
        resizeMode: 'cover',
        marginVertical: 10,
    },
});

export default Info;
